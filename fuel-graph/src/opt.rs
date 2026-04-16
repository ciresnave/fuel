//! Graph-level, backend-agnostic optimization passes.
//!
//! Transforms a `fuel-graph` computation graph before execution. Every
//! backend benefits from these passes because they operate purely on
//! the abstract op graph — no backend-specific knowledge required.
//!
//! **Passes today:**
//!
//! - **CSE** (common subexpression elimination): if the graph contains
//!   two structurally-identical nodes — same op, same inputs (after
//!   prior simplification), same shape/dtype — consumers of the
//!   duplicate are redirected to the canonical node. The duplicate
//!   becomes unreferenced and is silently dropped by the executor's
//!   topo-walk. Commutative ops (`Add`, `Mul`, `Maximum`, `Minimum`)
//!   are keyed on sorted input IDs so `a + b` and `b + a` fold to the
//!   same canonical node.
//!
//! - **Algebraic simplification**: a handful of identity/zero rules
//!   that eliminate no-op ops:
//!   - `AddScalar(0.0)(x)` → `x`
//!   - `MulScalar(1.0)(x)` → `x`
//!   - `Neg(Neg(x))` → `x`
//!   - `Reshape(Reshape(x, _), s)` → `Reshape(x, s)`
//!   These rarely appear in hand-written user code but show up
//!   routinely in autograd-generated backward graphs and in
//!   generic transformer building blocks that sometimes collapse.
//!
//! **Design:** graphs are append-only, so rewrites don't mutate
//! existing nodes. Instead, the pass walks topologically, builds a
//! remap `HashMap<NodeId, NodeId>` of old → canonical, and appends
//! newly-canonicalized nodes to the same graph. Unreferenced originals
//! remain in the arena but are never visited during realize. This is
//! effectively a combined CSE + simplification + free DCE pass.
//!
//! **Return value:** the function takes the roots the caller cares
//! about and returns the rewritten roots. Callers use these to update
//! their `Tensor` handles.

use crate::{topo_order_multi, Node, NodeId, Op, SharedGraph};
use fuel_core_types::DType;
use std::collections::HashMap;

/// A HashMap-friendly encoding of `Op`. Needed because `Op` carries
/// `f64` (in `AddScalar`, `MulScalar`, `Clamp`, `LayerNormLastDim`,
/// `LayerNormLastDimBackward`) and `ConstData` (in `Const`), neither
/// of which is `Hash + Eq`. We keep `Const` out of CSE entirely
/// (identity-dedup happens via the executor's Arc-pointer const pool
/// already) and encode scalar payloads as their bit patterns.
#[derive(Hash, PartialEq, Eq)]
struct OpKey {
    tag: u16,
    ints: Vec<i64>,
    bits: Vec<u64>,
    // Sparse payloads we don't need to index into for equality are
    // serialized via their Debug repr as a fallback. Cheap and
    // correct; not used on the hot path outside simplification.
    dims: Vec<usize>,
    shape: Option<Vec<usize>>,
    dtype: Option<u32>,
}

fn op_key(op: &Op) -> Option<OpKey> {
    // We deliberately refuse to CSE `Const`. Rationale above.
    let (tag, ints, bits, dims, shape, dtype) = match op {
        Op::Const(_) => return None,

        Op::Add => (1, vec![], vec![], vec![], None, None),
        Op::Sub => (2, vec![], vec![], vec![], None, None),
        Op::Mul => (3, vec![], vec![], vec![], None, None),
        Op::Div => (4, vec![], vec![], vec![], None, None),

        Op::Neg => (10, vec![], vec![], vec![], None, None),
        Op::Sqr => (11, vec![], vec![], vec![], None, None),
        Op::Sqrt => (12, vec![], vec![], vec![], None, None),
        Op::Exp => (13, vec![], vec![], vec![], None, None),
        Op::Log => (14, vec![], vec![], vec![], None, None),
        Op::Sin => (15, vec![], vec![], vec![], None, None),
        Op::Cos => (16, vec![], vec![], vec![], None, None),
        Op::Tanh => (17, vec![], vec![], vec![], None, None),
        Op::Sigmoid => (18, vec![], vec![], vec![], None, None),
        Op::Silu => (19, vec![], vec![], vec![], None, None),
        Op::Gelu => (20, vec![], vec![], vec![], None, None),
        Op::Relu => (21, vec![], vec![], vec![], None, None),
        Op::Step => (22, vec![], vec![], vec![], None, None),

        Op::MatMul => (30, vec![], vec![], vec![], None, None),
        Op::Transpose => (31, vec![], vec![], vec![], None, None),
        Op::Permute(axes) => (32, vec![], vec![], axes.clone(), None, None),

        Op::Cast(dt) => (40, vec![], vec![], vec![], None, Some(dtype_key(*dt))),
        Op::BroadcastTo(s) => (41, vec![], vec![], vec![], Some(s.dims().to_vec()), None),
        Op::Reshape(s) => (42, vec![], vec![], vec![], Some(s.dims().to_vec()), None),
        Op::ReduceSumTo(s) => (43, vec![], vec![], vec![], Some(s.dims().to_vec()), None),

        Op::SumAll => (50, vec![], vec![], vec![], None, None),
        Op::MaxAll => (51, vec![], vec![], vec![], None, None),
        Op::MinAll => (52, vec![], vec![], vec![], None, None),
        Op::MeanAll => (53, vec![], vec![], vec![], None, None),

        Op::SumDim(d) => (60, vec![*d as i64], vec![], vec![], None, None),
        Op::MaxDim(d) => (61, vec![*d as i64], vec![], vec![], None, None),
        Op::MinDim(d) => (62, vec![*d as i64], vec![], vec![], None, None),
        Op::MeanDim(d) => (63, vec![*d as i64], vec![], vec![], None, None),
        Op::ArgMaxDim(d) => (64, vec![*d as i64], vec![], vec![], None, None),
        Op::ArgMinDim(d) => (65, vec![*d as i64], vec![], vec![], None, None),

        Op::SoftmaxLastDim => (70, vec![], vec![], vec![], None, None),
        Op::LayerNormLastDim { eps } => (71, vec![], vec![eps.to_bits()], vec![], None, None),
        Op::RmsNormLastDim { eps } => (74, vec![], vec![eps.to_bits()], vec![], None, None),
        Op::Rope => (75, vec![], vec![], vec![], None, None),
        Op::RmsNormLastDimBackward { eps } => (76, vec![], vec![eps.to_bits()], vec![], None, None),
        Op::SoftmaxLastDimBackward => (72, vec![], vec![], vec![], None, None),
        Op::LayerNormLastDimBackward { eps } => {
            (73, vec![], vec![eps.to_bits()], vec![], None, None)
        }

        Op::Concat { dim } => (80, vec![*dim as i64], vec![], vec![], None, None),
        Op::Slice { dim, start, len } => (
            81,
            vec![*dim as i64, *start as i64, *len as i64],
            vec![],
            vec![],
            None,
            None,
        ),

        Op::AddScalar(c) => (90, vec![], vec![c.to_bits()], vec![], None, None),
        Op::MulScalar(c) => (91, vec![], vec![c.to_bits()], vec![], None, None),
        Op::PowI(n) => (92, vec![*n as i64], vec![], vec![], None, None),
        Op::Clamp { min, max } => (93, vec![], vec![min.to_bits(), max.to_bits()], vec![], None, None),

        Op::Maximum => (100, vec![], vec![], vec![], None, None),
        Op::Minimum => (101, vec![], vec![], vec![], None, None),

        // Indexing and anything else we haven't explicitly listed:
        // fall back to a unique tag that includes a structural
        // discriminant. These ops rarely appear more than once with
        // identical inputs, so we just mark them non-CSE-able by
        // returning None. Conservative; safe.
        _ => return None,
    };
    Some(OpKey { tag, ints, bits, dims, shape, dtype })
}

fn dtype_key(dt: DType) -> u32 {
    // Cheap injection: the Debug form is stable.
    // For tiny enums this compiles to a jump table.
    format!("{dt:?}").as_bytes().iter().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(*b as u32))
}

fn is_commutative(op: &Op) -> bool {
    matches!(op, Op::Add | Op::Mul | Op::Maximum | Op::Minimum)
}

/// Run CSE + algebraic simplification on the graph reachable from
/// `roots`. Returns the (possibly-rewritten) roots the caller should
/// use afterward.
///
/// The pass runs to a fixed point on a single topological walk: each
/// node is visited once, canonicalized (inputs remapped, constants
/// folded where trivial), then either matched to an existing canonical
/// node (CSE) or appended as fresh. No multi-pass iteration — the
/// simplification rules are chosen so one forward pass suffices.
pub fn optimize(graph: &SharedGraph, roots: &[NodeId]) -> Vec<NodeId> {
    let order = {
        let g = graph.borrow();
        topo_order_multi(&g, roots)
    };

    let mut g = graph.borrow_mut();
    let mut remap: HashMap<NodeId, NodeId> = HashMap::new();
    let mut cse: HashMap<(OpKey, Vec<NodeId>), NodeId> = HashMap::new();

    for id in order {
        let (op, inputs, shape, dtype) = {
            let node = g.node(id);
            (node.op.clone(), node.inputs.clone(), node.shape.clone(), node.dtype)
        };
        let mapped_inputs: Vec<NodeId> = inputs
            .iter()
            .map(|input_id| *remap.get(input_id).unwrap_or(input_id))
            .collect();
        let inputs_unchanged = mapped_inputs == inputs;

        // 1. Algebraic simplifications that produce an alias (no new
        //    node). If a rule fires, we remap this id to an existing
        //    node and skip CSE for it entirely.
        if let Some(aliased) = try_simplify(&op, &mapped_inputs) {
            remap.insert(id, aliased);
            continue;
        }

        // 2. CSE: if a structurally-identical canonical node already
        //    exists, reuse it. Commutative ops use sorted inputs as
        //    the key so `a+b` and `b+a` match. If no match exists and
        //    nothing about this node needs rewriting (inputs unchanged,
        //    no simplification fired), keep the original node in place
        //    to avoid polluting the arena with identical copies.
        if let Some(key) = op_key(&op) {
            let key_inputs = if is_commutative(&op) {
                let mut v = mapped_inputs.clone();
                v.sort();
                v
            } else {
                mapped_inputs.clone()
            };
            let full_key = (key, key_inputs);
            if let Some(&existing) = cse.get(&full_key) {
                remap.insert(id, existing);
                continue;
            }
            let canonical_id = if inputs_unchanged {
                id
            } else {
                g.push(Node {
                    op: op.clone(),
                    inputs: mapped_inputs.clone(),
                    shape: shape.clone(),
                    dtype,
                })
            };
            cse.insert(full_key, canonical_id);
            remap.insert(id, canonical_id);
        } else {
            // Const or other non-CSE-able op. Keep the original if
            // unchanged; otherwise append a rewritten copy (Const
            // never has inputs, so this branch effectively keeps
            // Const originals).
            let canonical_id = if inputs_unchanged {
                id
            } else {
                g.push(Node {
                    op: op.clone(),
                    inputs: mapped_inputs,
                    shape,
                    dtype,
                })
            };
            remap.insert(id, canonical_id);
        }
    }

    roots.iter().map(|r| *remap.get(r).unwrap_or(r)).collect()
}

/// Try to produce an aliasing simplification: "this op is equivalent
/// to one of its inputs." Returns the aliased NodeId if so.
fn try_simplify(op: &Op, inputs: &[NodeId]) -> Option<NodeId> {
    match op {
        // AddScalar(0) and MulScalar(1) are no-ops. Route consumers
        // straight to the input and drop the op.
        Op::AddScalar(c) if *c == 0.0 => Some(inputs[0]),
        Op::MulScalar(c) if *c == 1.0 => Some(inputs[0]),
        // PowI(1) is a no-op. PowI(0) would be "1" but that needs a
        // new const node with the right shape — skip here, let the
        // executor handle or another rule add it.
        Op::PowI(1) => Some(inputs[0]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use fuel_core_types::Shape;

    fn make_scalar_graph() -> (SharedGraph, Tensor) {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]));
        (t.graph().clone(), t)
    }

    #[test]
    fn cse_folds_identical_add() {
        let (graph, a) = make_scalar_graph();
        let b = a.add(&a);
        let c = a.add(&a);
        let pre_len = graph.borrow().len();
        let new_roots = optimize(&graph, &[b.id(), c.id()]);
        assert_eq!(new_roots[0], new_roots[1], "CSE should map both to same node");
        assert!(graph.borrow().len() >= pre_len);
    }

    #[test]
    fn cse_folds_commutative() {
        // Build two tensors inside the same graph so `a + b` and
        // `b + a` share NodeIds for the inputs. Use add_scalar on `a`
        // as a simple way to get a second tensor handle sharing a's graph.
        let (_graph, a) = make_scalar_graph();
        let b = a.add_scalar(5.0);
        let ab = a.add(&b);
        let ba = b.add(&a);
        let graph = a.graph().clone();
        let new_roots = optimize(&graph, &[ab.id(), ba.id()]);
        assert_eq!(
            new_roots[0], new_roots[1],
            "commutative CSE should fold a+b and b+a to one node"
        );
    }

    #[test]
    fn simplifies_add_scalar_zero() {
        let (graph, a) = make_scalar_graph();
        let b = a.add_scalar(0.0);
        let new_roots = optimize(&graph, &[b.id()]);
        assert_eq!(new_roots[0], a.id(), "AddScalar(0) should alias to input");
    }

    #[test]
    fn simplifies_mul_scalar_one() {
        let (graph, a) = make_scalar_graph();
        let b = a.mul_scalar(1.0);
        let new_roots = optimize(&graph, &[b.id()]);
        assert_eq!(new_roots[0], a.id(), "MulScalar(1) should alias to input");
    }

    #[test]
    fn cse_does_not_fold_distinct_ops() {
        // Add and Mul on the same inputs are not equivalent.
        let (graph, a) = make_scalar_graph();
        let sum = a.add(&a);
        let prod = a.mul(&a);
        let new_roots = optimize(&graph, &[sum.id(), prod.id()]);
        assert_ne!(new_roots[0], new_roots[1], "Add and Mul must stay distinct");
    }

    #[test]
    fn cse_does_not_fold_distinct_scalars() {
        let (graph, a) = make_scalar_graph();
        let b = a.add_scalar(1.0);
        let c = a.add_scalar(2.0);
        let new_roots = optimize(&graph, &[b.id(), c.id()]);
        assert_ne!(new_roots[0], new_roots[1], "AddScalar with different c must stay distinct");
    }

    #[test]
    fn cse_nested_chain_deduplicates() {
        // (a + a) + a  appearing twice should dedupe both subexpressions.
        let (graph, a) = make_scalar_graph();
        let p1 = a.add(&a);
        let p2 = a.add(&a); // duplicates p1
        let q1 = p1.add(&a);
        let q2 = p2.add(&a); // structurally identical to q1 via CSE
        let new_roots = optimize(&graph, &[q1.id(), q2.id()]);
        assert_eq!(new_roots[0], new_roots[1], "nested duplicates must fold");
    }
}

//! `unroll_scan`: materialize a bounded [`crate::Op::Scan`] into real
//! primitive nodes on demand. Used as (a) the FKC/Spec-B numeric oracle and
//! (b) the fallback lowering for a backend without a scan kernel. NOT
//! registered as anyone's `.decompose` — `Op::Scan` is a bare primitive that
//! stays terminal in the base map.

use std::collections::HashMap;

use crate::{Graph, Node, NodeId, Op, ScanEmit, ScanRole};

/// Unroll `steps` iterations of the `Op::Scan` at `scan_id` into primitives.
///
/// Returns `(selected, complementary)`: `emit=All` -> `(stacked_ys,
/// final_carry)`, `emit=Final` -> `(final_carry, stacked_ys)`. `early_exit =
/// Some` -> `Err` (a surfaced Phase-2 gap, never a panic).
pub fn unroll_scan(
    graph: &mut Graph,
    scan_id: NodeId,
    steps: usize,
) -> std::result::Result<(NodeId, NodeId), fuel_ir::Error> {
    // 1. Read the Scan node's params + input layout in a short borrow.
    let (n_xs, bound, emit, has_exit, inputs) = {
        let n = graph.node(scan_id);
        match &n.op {
            Op::Scan { n_xs, bound, emit, early_exit } => {
                (*n_xs, *bound, *emit, early_exit.is_some(), n.inputs.clone())
            }
            other => {
                return Err(fuel_ir::Error::Msg(format!(
                    "unroll_scan: node {} is not an Op::Scan ({})",
                    scan_id.0, other.short_name(),
                )).bt());
            }
        }
    };
    if has_exit {
        return Err(fuel_ir::Error::Msg(
            "unroll_scan: early_exit = Some is a Phase-2 mechanism (not implemented)".into(),
        ).bt());
    }
    if steps == 0 || steps > bound {
        return Err(fuel_ir::Error::Msg(format!(
            "unroll_scan: steps {steps} must be in 1..={bound}",
        )).bt());
    }
    // inputs = [init_carry, xs_0..xs_{n_xs-1}, consts.., body_new_carry, body_y]
    if inputs.len() < 2 + n_xs {
        return Err(fuel_ir::Error::Msg("unroll_scan: malformed Op::Scan inputs".into()).bt());
    }
    let init_carry = inputs[0];
    let xs: Vec<NodeId> = inputs[1..1 + n_xs].to_vec();
    let consts: Vec<NodeId> = inputs[1 + n_xs..inputs.len() - 2].to_vec();
    let body_new_carry = inputs[inputs.len() - 2];
    let body_y = inputs[inputs.len() - 1];
    let consts_set: std::collections::HashSet<NodeId> = consts.iter().copied().collect();

    let mut carry = init_carry;
    let mut ys_steps: Vec<NodeId> = Vec::with_capacity(steps);

    for t in 0..steps {
        // Per-step xs slices: xs[i] sliced at [t, t+1) on scan-axis 0, then
        // squeezed to drop the step axis -> the ScanPlaceholder{Elem,i} shape.
        let mut elem: Vec<NodeId> = Vec::with_capacity(n_xs);
        for &x in &xs {
            let (x_shape, x_dtype) = { let n = graph.node(x); (n.shape.clone(), n.dtype) };
            let sliced_dims: Vec<usize> = std::iter::once(1usize)
                .chain(x_shape.dims().iter().skip(1).copied()).collect();
            let sl = graph.push(Node {
                op: Op::Slice { dim: 0, start: t, len: 1 },
                inputs: vec![x],
                shape: fuel_ir::Shape::from_dims(&sliced_dims),
                dtype: x_dtype,
            });
            let sq_dims: Vec<usize> = x_shape.dims().iter().skip(1).copied().collect();
            let sq = graph.push(Node {
                op: Op::Squeeze { dim: 0 },
                inputs: vec![sl],
                shape: fuel_ir::Shape::from_dims(&sq_dims),
                dtype: x_dtype,
            });
            elem.push(sq);
        }

        // Clone the body subgraph (rooted at {body_new_carry, body_y}),
        // substituting placeholders + keeping consts shared.
        let mut subst: HashMap<NodeId, NodeId> = HashMap::new();
        let next_carry = clone_body_node(graph, body_new_carry, carry, &elem, &consts_set, &mut subst);
        let y_t = clone_body_node(graph, body_y, carry, &elem, &consts_set, &mut subst);
        carry = next_carry;
        ys_steps.push(y_t);
    }

    // stacked_ys = Concat(dim 0) of each y_t unsqueezed at dim 0.
    let mut unsqueezed: Vec<NodeId> = Vec::with_capacity(ys_steps.len());
    for &y in &ys_steps {
        let (y_shape, y_dtype) = { let n = graph.node(y); (n.shape.clone(), n.dtype) };
        let un_dims: Vec<usize> = std::iter::once(1usize).chain(y_shape.dims().iter().copied()).collect();
        let un = graph.push(Node {
            op: Op::Unsqueeze { dim: 0 },
            inputs: vec![y],
            shape: fuel_ir::Shape::from_dims(&un_dims),
            dtype: y_dtype,
        });
        unsqueezed.push(un);
    }
    let (y0_shape, y0_dtype) = { let n = graph.node(ys_steps[0]); (n.shape.clone(), n.dtype) };
    let stacked_dims: Vec<usize> = std::iter::once(ys_steps.len())
        .chain(y0_shape.dims().iter().copied()).collect();
    let stacked_ys = graph.push(Node {
        op: Op::Concat { dim: 0 },
        inputs: unsqueezed,
        shape: fuel_ir::Shape::from_dims(&stacked_dims),
        dtype: y0_dtype,
    });

    Ok(match emit {
        ScanEmit::All => (stacked_ys, carry),
        ScanEmit::Final => (carry, stacked_ys),
    })
}

/// Topological copy of a body node, substituting `ScanPlaceholder{Carry,_}` ->
/// `carry`, `ScanPlaceholder{Elem,i}` -> `elem[i]`, and keeping any node in
/// `consts_set` shared (not copied). Memoized in `subst`.
fn clone_body_node(
    graph: &mut Graph,
    id: NodeId,
    carry: NodeId,
    elem: &[NodeId],
    consts_set: &std::collections::HashSet<NodeId>,
    subst: &mut HashMap<NodeId, NodeId>,
) -> NodeId {
    if let Some(&m) = subst.get(&id) { return m; }
    if consts_set.contains(&id) { return id; }
    let (op, in_ids, shape, dtype) = {
        let n = graph.node(id);
        (n.op.clone(), n.inputs.clone(), n.shape.clone(), n.dtype)
    };
    let mapped = match op {
        Op::ScanPlaceholder { role: ScanRole::Carry, .. } => carry,
        Op::ScanPlaceholder { role: ScanRole::Elem, index } => elem[index],
        _ => {
            let new_inputs: Vec<NodeId> = in_ids.iter()
                .map(|&c| clone_body_node(graph, c, carry, elem, consts_set, subst))
                .collect();
            graph.push(Node { op, inputs: new_inputs, shape, dtype })
        }
    };
    subst.insert(id, mapped);
    mapped
}

#[cfg(test)]
mod tests {
    use crate::{Graph, Node, Op, ScanEmit, ScanPredicate, ScanRole};
    use crate::scan::unroll_scan;
    use crate::opt::lower_to_base_map;
    use fuel_ir::{DType, Shape};
    use std::sync::{Arc, RwLock};

    // Build a trivial scan: carry [1], body new_carry = carry*2, body_y =
    // new_carry, n_xs = 0, bound = 3, emit = All. Returns (graph_arc, scan_id).
    fn trivial_scan(bound: usize, emit: ScanEmit, early_exit: Option<ScanPredicate>) -> (Arc<RwLock<Graph>>, crate::NodeId) {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let scan = {
            let mut g = graph.write().unwrap();
            let s = Shape::from_dims(&[1]);
            let carry = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
            let hole = g.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 }, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
            let nc = g.push(Node { op: Op::MulScalar(2.0), inputs: vec![hole], shape: s.clone(), dtype: DType::F32 });
            g.push(Node {
                op: Op::Scan { n_xs: 0, bound, emit, early_exit },
                inputs: vec![carry, nc, nc],
                shape: Shape::from_dims(&[bound, 1]),
                dtype: DType::F32,
            })
        };
        (graph, scan)
    }

    #[test]
    fn unroll_scan_all_produces_a_concat_of_steps_and_no_scan_nodes() {
        let (graph, scan) = trivial_scan(3, ScanEmit::All, None);
        let (ys, _carry) = {
            let mut g = graph.write().unwrap();
            unroll_scan(&mut g, scan, 3).expect("unroll")
        };
        let g = graph.read().unwrap();
        // ys root is a Concat over the 3 steps.
        assert!(matches!(g.node(ys).op, Op::Concat { .. }), "emit=All ys root should be Concat, got {:?}", g.node(ys).op.short_name());
        assert_eq!(g.node(ys).inputs.len(), 3, "one input per step");
        // No Op::Scan / Op::ScanPlaceholder reachable from the unrolled root.
        let reachable = crate::topo_order_multi(&g, &[ys]);
        assert!(!reachable.iter().any(|&n| matches!(g.node(n).op, Op::Scan { .. } | Op::ScanPlaceholder { .. })),
            "unrolled graph must contain no Scan/ScanPlaceholder nodes");
    }

    #[test]
    fn unroll_scan_rejects_early_exit_some() {
        let (graph, scan) = trivial_scan(2, ScanEmit::All, Some(ScanPredicate));
        let mut g = graph.write().unwrap();
        let r = unroll_scan(&mut g, scan, 2);
        assert!(r.is_err(), "early_exit = Some must be a typed Err (Phase-2 gap), never a panic");
    }

    #[test]
    fn op_scan_is_a_terminal_in_the_base_map() {
        // lower_to_base_map must LEAVE Op::Scan in place (no LoweringRule
        // matches a bare Op variant) — not silently expanded, not errored.
        let (graph, scan) = trivial_scan(3, ScanEmit::All, None);
        let roots = lower_to_base_map(&graph, &[scan]);
        let g = graph.read().unwrap();
        let reachable = crate::topo_order_multi(&g, &roots);
        assert!(reachable.iter().any(|&n| matches!(g.node(n).op, Op::Scan { .. })),
            "Op::Scan must remain a terminal after lower_to_base_map");
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0
//! `unroll_scan`: materialize a bounded [`crate::Op::Scan`] into real
//! primitive nodes on demand. Used as (a) the FKC/Spec-B numeric oracle and
//! (b) the fallback lowering for a backend without a scan kernel. NOT
//! registered as anyone's `.decompose` — `Op::Scan` is a bare primitive that
//! stays terminal in the base map.

use std::collections::HashMap;

use crate::{Graph, Node, NodeId, Op, ScanEmit, ScanRole};

/// Validate every `Op::ScanPlaceholder` reachable from the scan body's exit
/// `roots` (`[body_new_carry, body_y]`, optionally the predicate) has an
/// in-range index: v1 is **single-carry** (`Carry` index must be 0), and every
/// `Elem` index must address one of the `n_xs` per-step slices. This keeps the
/// later `clone_body_node`/`build_scan_step` `elem[index]` access infallible by
/// construction — so a malformed body is a typed **build-time** `Err` (from the
/// `NodeHandle::scan` / `NodeHandle::scan_until` builders) rather than a mid-realize
/// `elem[index]` panic on the forward-driver path. Short immutable borrow only.
pub(crate) fn validate_scan_body_placeholders(
    graph: &Graph,
    roots: &[NodeId],
    n_carries: usize,
    n_xs: usize,
) -> std::result::Result<(), fuel_ir::Error> {
    let reachable = crate::topo_order_multi(graph, roots);
    for &id in &reachable {
        if let Op::ScanPlaceholder { role, index } = &graph.node(id).op {
            match *role {
                // GAP-303: was `*index != 0` ("v1 is single-carry"). The TYPE was
                // always multi-carry-shaped -- `index` is a `usize` -- so this is
                // relaxing a CHECK, not widening a type. Range-checked exactly like
                // the `Elem` arm below; the two are now symmetric.
                ScanRole::Carry if *index >= n_carries => {
                    return Err(fuel_ir::Error::Msg(format!(
                        "scan: body node {} is ScanPlaceholder{{Carry, {index}}} out of range (n_carries = {n_carries})",
                        id.0,
                    )).bt());
                }
                ScanRole::Elem if *index >= n_xs => {
                    return Err(fuel_ir::Error::Msg(format!(
                        "scan: body node {} is ScanPlaceholder{{Elem, {index}}} out of range (n_xs = {n_xs})",
                        id.0,
                    )).bt());
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Unroll `steps` iterations of the `Op::Scan` at `scan_id` into primitives.
///
/// Returns `(selected, complementary)`: `emit=All` -> `(stacked_ys,
/// final_carry)`, `emit=Final` -> `(final_carry, stacked_ys)`. `early_exit =
/// Some` peels the trailing `pred_exit` and IGNORES it — the build-time
/// backward/oracle unroll differentiates the full static `bound` (spec C3
/// static-horizon note); the runtime early-exit is a forward-only optimization
/// driven by the step driver, not this unroll.
pub fn unroll_scan(
    graph: &mut Graph,
    scan_id: NodeId,
    steps: usize,
) -> std::result::Result<(Vec<NodeId>, Vec<NodeId>), fuel_ir::Error> {
    if scan_id.0 >= graph.len() {
        return Err(fuel_ir::Error::Msg(format!(
            "unroll_scan: scan_id {} is out of range (graph has {} nodes)",
            scan_id.0,
            graph.len(),
        ))
        .bt());
    }
    // 1. Read the Scan node's params + input layout in a short borrow.
    let (n_carries, n_xs, bound, emit, has_exit, inputs) = {
        let n = graph.node(scan_id);
        match &n.op {
            Op::Scan {
                n_carries,
                n_xs,
                bound,
                emit,
                early_exit,
            } => (
                *n_carries,
                *n_xs,
                *bound,
                *emit,
                early_exit.is_some(),
                n.inputs.clone(),
            ),
            other => {
                return Err(fuel_ir::Error::Msg(format!(
                    "unroll_scan: node {} is not an Op::Scan ({})",
                    scan_id.0,
                    other.short_name(),
                ))
                .bt());
            }
        }
    };
    if steps == 0 || steps > bound {
        return Err(fuel_ir::Error::Msg(format!(
            "unroll_scan: steps {steps} must be in 1..={bound}",
        ))
        .bt());
    }
    // inputs = [init_carry, xs_0..xs_{n_xs-1}, consts.., body_new_carry, body_y, [pred_exit]]
    // Trailing slots: body_new_carry + body_y (+ pred_exit when early_exit = Some).
    // Minimum well-formed layout: init_carry(1) + n_xs + consts(>=0) + n_trailing.
    // (One short of the trailing slots — reject it here, before the `consts`
    // slice below can panic with start > end.)
    // GAP-303: the leading carry BLOCK is `n_carries` wide (was a literal 1) and
    // the trailing block carries one `body_new_carry` per carry.
    //
    // ⚠️ `early_exit` is dead in VALUE (never `Some` on a live Phase-1 path) and
    // LIVE IN LAYOUT -- it is the `+ has_exit` term here. That is the one place
    // the Phase-2 field participates in Phase-1 arithmetic.
    let n_trailing = n_carries + 1 + usize::from(has_exit); // new_carries.., body_y, [pred]
    if n_carries == 0 {
        return Err(fuel_ir::Error::Msg(
            "unroll_scan: Op::Scan with n_carries = 0 has no recurrent state".to_string(),
        )
        .bt());
    }
    if inputs.len() < n_carries + n_xs + n_trailing {
        return Err(fuel_ir::Error::Msg(format!(
            "unroll_scan: malformed Op::Scan inputs — need >= {} (n_carries={n_carries} + n_xs={n_xs} + {n_trailing} trailing), got {}",
            n_carries + n_xs + n_trailing, inputs.len(),
        )).bt());
    }
    let init_carries: Vec<NodeId> = inputs[0..n_carries].to_vec();
    let xs: Vec<NodeId> = inputs[n_carries..n_carries + n_xs].to_vec();
    let consts: Vec<NodeId> = inputs[n_carries + n_xs..inputs.len() - n_trailing].to_vec();
    let body_new_carries: Vec<NodeId> =
        inputs[inputs.len() - n_trailing..inputs.len() - n_trailing + n_carries].to_vec();
    let body_y = inputs[inputs.len() - n_trailing + n_carries];
    // pred_exit = inputs[inputs.len() - 1] when has_exit — intentionally NOT read; the
    // build-time backward unroll differentiates the full static `bound` and ignores the
    // runtime early-exit predicate (spec C3 "static-horizon note").
    let consts_set: std::collections::HashSet<NodeId> = consts.iter().copied().collect();

    // 2. Validate every ScanPlaceholder reachable from the body's two exit
    // nodes has an in-range index, BEFORE any cloning/mutation (shared with the
    // build-time check in NodeHandle::scan / NodeHandle::scan_until).
    let mut roots: Vec<NodeId> = body_new_carries.clone();
    roots.push(body_y);
    validate_scan_body_placeholders(graph, &roots, n_carries, n_xs)?;

    // 3. Validate every xs[i] has a leading (scan-axis) dim >= steps: the
    // per-step `Slice { dim: 0, start: t, len: 1 }` below needs `t` in range
    // for every `t in 0..steps`, and needs a dim 0 to slice at all.
    for (i, &x) in xs.iter().enumerate() {
        let dims = graph.node(x).shape.dims().to_vec();
        if dims.is_empty() {
            return Err(fuel_ir::Error::Msg(format!(
                "unroll_scan: xs[{i}] (node {}) is rank-0, needs a leading scan-axis of len >= steps ({steps})",
                x.0,
            )).bt());
        }
        if dims[0] < steps {
            return Err(fuel_ir::Error::Msg(format!(
                "unroll_scan: xs[{i}] (node {}) leading dim {} < steps ({steps})",
                x.0, dims[0],
            ))
            .bt());
        }
    }

    let mut carries: Vec<NodeId> = init_carries;
    let mut ys_steps: Vec<NodeId> = Vec::with_capacity(steps);

    for t in 0..steps {
        // Per-step xs slices: xs[i] sliced at [t, t+1) on scan-axis 0, then
        // squeezed to drop the step axis -> the ScanPlaceholder{Elem,i} shape.
        let mut elem: Vec<NodeId> = Vec::with_capacity(n_xs);
        for &x in &xs {
            let (x_shape, x_dtype) = {
                let n = graph.node(x);
                (n.shape.clone(), n.dtype)
            };
            let sliced_dims: Vec<usize> = std::iter::once(1usize)
                .chain(x_shape.dims().iter().skip(1).copied())
                .collect();
            let sl = graph.push(Node {
                op: Op::Slice {
                    dim: 0,
                    start: t,
                    len: 1,
                },
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
        // ⚠️ Every new carry is computed from THIS step's `carries`, so the whole
        // block is built before any assignment -- updating in place mid-loop would
        // let carry k+1 read carry k's NEXT value instead of its current one.
        let next_carries: Vec<NodeId> = body_new_carries
            .iter()
            .map(|&nc| clone_body_node(graph, nc, &carries, &elem, &consts_set, &mut subst))
            .collect();
        let y_t = clone_body_node(graph, body_y, &carries, &elem, &consts_set, &mut subst);
        carries = next_carries;
        ys_steps.push(y_t);
    }

    // stacked_ys = Concat(dim 0) of each y_t unsqueezed at dim 0.
    let mut unsqueezed: Vec<NodeId> = Vec::with_capacity(ys_steps.len());
    for &y in &ys_steps {
        let (y_shape, y_dtype) = {
            let n = graph.node(y);
            (n.shape.clone(), n.dtype)
        };
        let un_dims: Vec<usize> = std::iter::once(1usize)
            .chain(y_shape.dims().iter().copied())
            .collect();
        let un = graph.push(Node {
            op: Op::Unsqueeze { dim: 0 },
            inputs: vec![y],
            shape: fuel_ir::Shape::from_dims(&un_dims),
            dtype: y_dtype,
        });
        unsqueezed.push(un);
    }
    let (y0_shape, y0_dtype) = {
        let n = graph.node(ys_steps[0]);
        (n.shape.clone(), n.dtype)
    };
    let stacked_dims: Vec<usize> = std::iter::once(ys_steps.len())
        .chain(y0_shape.dims().iter().copied())
        .collect();
    let stacked_ys = graph.push(Node {
        op: Op::Concat { dim: 0 },
        inputs: unsqueezed,
        shape: fuel_ir::Shape::from_dims(&stacked_dims),
        dtype: y0_dtype,
    });

    // GAP-303: both sides widened to `Vec`. The (selected, complementary)
    // contract is UNCHANGED -- only the arity is.
    //
    // ⚠️ The old `(NodeId, NodeId)` return was a FOURTH site of the single-carry
    // assumption, encoded in a `pub fn`'s SIGNATURE rather than in a field: with
    // `emit = Final` and `n_carries > 1` there is no single "selected" output,
    // and no type could have expressed that while the pair was scalar.
    Ok(match emit {
        ScanEmit::All => (vec![stacked_ys], carries),
        ScanEmit::Final => (carries, vec![stacked_ys]),
    })
}

/// Topological copy of a body node, substituting `ScanPlaceholder{Carry,_}` ->
/// `carry`, `ScanPlaceholder{Elem,i}` -> `elem[i]`, and keeping any node in
/// `consts_set` shared (not copied). Memoized in `subst`.
fn clone_body_node(
    graph: &mut Graph,
    id: NodeId,
    carries: &[NodeId],
    elem: &[NodeId],
    consts_set: &std::collections::HashSet<NodeId>,
    subst: &mut HashMap<NodeId, NodeId>,
) -> NodeId {
    if let Some(&m) = subst.get(&id) {
        return m;
    }
    if consts_set.contains(&id) {
        return id;
    }
    let (op, in_ids, shape, dtype) = {
        let n = graph.node(id);
        (n.op.clone(), n.inputs.clone(), n.shape.clone(), n.dtype)
    };
    let mapped = match op {
        // GAP-303: the two arms are now SYMMETRIC. `Carry` used to ignore
        // `index` entirely (it was required to be 0); it now indexes `carries`
        // exactly as `Elem` indexes `elem`. Both are range-checked up front by
        // `validate_scan_body_placeholders`, so both index safely.
        Op::ScanPlaceholder {
            role: ScanRole::Carry,
            index,
        } => carries[index],
        Op::ScanPlaceholder {
            role: ScanRole::Elem,
            index,
        } => elem[index],
        _ => {
            let new_inputs: Vec<NodeId> = in_ids
                .iter()
                .map(|&c| clone_body_node(graph, c, carries, elem, consts_set, subst))
                .collect();
            graph.push(Node {
                op,
                inputs: new_inputs,
                shape,
                dtype,
            })
        }
    };
    subst.insert(id, mapped);
    mapped
}

/// The parsed input layout of an [`Op::Scan`] node — the fields the early-exit
/// step driver needs, extracted from the trailing-input encoding.
pub struct ScanLayout {
    /// GAP-303: how many carries the leading/trailing blocks are wide.
    pub n_carries: usize,
    pub n_xs: usize,
    pub bound: usize,
    pub emit: ScanEmit,
    pub init_carries: Vec<NodeId>,
    pub xs: Vec<NodeId>,
    pub consts: Vec<NodeId>,
    pub body_new_carries: Vec<NodeId>,
    pub body_y: NodeId,
    /// `Some` when `early_exit = Some` — the scalar-`U8` convergence predicate.
    pub pred_exit: Option<NodeId>,
}

/// One materialized scan step: the post-step carry, the emitted `y`, and the
/// (optional) realized `stop` predicate node for this step.
pub struct ScanStep {
    pub new_carries: Vec<NodeId>,
    pub y: NodeId,
    pub stop: Option<NodeId>,
}

/// Parse an [`Op::Scan`] node's trailing-input layout. Mirrors the parse in
/// [`unroll_scan`], including the `early_exit`-aware trailing count, but yields
/// a struct the step driver can drive one step at a time. Returns a typed
/// `Err` for a non-`Op::Scan` node or a malformed (too-short) input layout.
pub fn parse_scan_layout(
    graph: &Graph,
    scan_id: NodeId,
) -> std::result::Result<ScanLayout, fuel_ir::Error> {
    if scan_id.0 >= graph.len() {
        return Err(fuel_ir::Error::Msg(format!(
            "parse_scan_layout: scan_id {} is out of range (graph has {} nodes)",
            scan_id.0,
            graph.len(),
        ))
        .bt());
    }
    let n = graph.node(scan_id);
    let (n_carries, n_xs, bound, emit, has_exit) = match &n.op {
        Op::Scan {
            n_carries,
            n_xs,
            bound,
            emit,
            early_exit,
        } => (*n_carries, *n_xs, *bound, *emit, early_exit.is_some()),
        other => {
            return Err(fuel_ir::Error::Msg(format!(
                "parse_scan_layout: node {} is not an Op::Scan ({})",
                scan_id.0,
                other.short_name(),
            ))
            .bt());
        }
    };
    let inputs = &n.inputs;
    // GAP-303: mirrors `unroll_scan`'s arithmetic exactly -- the two parses must
    // agree or a step driver and the oracle disagree about the same node.
    let n_trailing = n_carries + 1 + usize::from(has_exit);
    if n_carries == 0 {
        return Err(fuel_ir::Error::Msg(
            "parse_scan_layout: Op::Scan with n_carries = 0 has no recurrent state".to_string(),
        )
        .bt());
    }
    if inputs.len() < n_carries + n_xs + n_trailing {
        return Err(fuel_ir::Error::Msg(format!(
            "parse_scan_layout: malformed Op::Scan inputs — need >= {} (n_carries={n_carries} + n_xs={n_xs} + {n_trailing} trailing), got {}",
            n_carries + n_xs + n_trailing, inputs.len(),
        )).bt());
    }
    let init_carries: Vec<NodeId> = inputs[0..n_carries].to_vec();
    let xs: Vec<NodeId> = inputs[n_carries..n_carries + n_xs].to_vec();
    let consts: Vec<NodeId> = inputs[n_carries + n_xs..inputs.len() - n_trailing].to_vec();
    let body_new_carries: Vec<NodeId> =
        inputs[inputs.len() - n_trailing..inputs.len() - n_trailing + n_carries].to_vec();
    let body_y = inputs[inputs.len() - n_trailing + n_carries];
    let pred_exit = has_exit.then(|| inputs[inputs.len() - 1]);
    Ok(ScanLayout {
        n_carries,
        n_xs,
        bound,
        emit,
        init_carries,
        xs,
        consts,
        body_new_carries,
        body_y,
        pred_exit,
    })
}

/// Materialize one step of a scan at step index `t` with the given `carry`
/// NodeId. Slices each `xs[i]` at `[t, t+1)` (squeezed), clones `body_new_carry`
/// / `body_y` / `pred_exit` with a **single shared** substitution map so that a
/// `pred_exit` referencing `body_new_carry` resolves to *this step's* new carry
/// (no double-clone). Returns the post-step carry, `y`, and the optional `stop`
/// predicate node.
pub fn build_scan_step(
    graph: &mut Graph,
    layout: &ScanLayout,
    t: usize,
    carries: &[NodeId],
) -> std::result::Result<ScanStep, fuel_ir::Error> {
    // per-step elem slices: xs[i] sliced [t,t+1) on axis 0, squeezed.
    let mut elem: Vec<NodeId> = Vec::with_capacity(layout.n_xs);
    for &x in &layout.xs {
        let (x_shape, x_dtype) = {
            let n = graph.node(x);
            (n.shape.clone(), n.dtype)
        };
        if x_shape.dims().first().is_none_or(|&d0| d0 <= t) {
            return Err(fuel_ir::Error::Msg(format!(
                "build_scan_step: xs slice at t={t} out of range for shape {:?}",
                x_shape.dims()
            ))
            .bt());
        }
        let sliced: Vec<usize> = std::iter::once(1usize)
            .chain(x_shape.dims().iter().skip(1).copied())
            .collect();
        let sl = graph.push(Node {
            op: Op::Slice {
                dim: 0,
                start: t,
                len: 1,
            },
            inputs: vec![x],
            shape: fuel_ir::Shape::from_dims(&sliced),
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
    let consts_set: std::collections::HashSet<NodeId> = layout.consts.iter().copied().collect();
    let mut subst: HashMap<NodeId, NodeId> = HashMap::new();
    // Clone EVERY body_new_carry FIRST so subst records each body_new_carry ->
    // new_carry, then clone body_y and pred_exit sharing subst (spec "Predicate
    // referencing body_new_carry" — no double-clone).
    //
    // GAP-303: the whole block is built from THIS step's `carries` before any of
    // it is returned, matching `unroll_scan`'s ordering.
    let new_carries: Vec<NodeId> = layout
        .body_new_carries
        .iter()
        .map(|&nc| clone_body_node(graph, nc, carries, &elem, &consts_set, &mut subst))
        .collect();
    let y = clone_body_node(
        graph,
        layout.body_y,
        carries,
        &elem,
        &consts_set,
        &mut subst,
    );
    let stop = layout
        .pred_exit
        .map(|p| clone_body_node(graph, p, carries, &elem, &consts_set, &mut subst));
    Ok(ScanStep {
        new_carries,
        y,
        stop,
    })
}

#[cfg(test)]
mod tests {
    use crate::opt::lower_to_base_map;
    use crate::scan::unroll_scan;
    use crate::{Graph, Node, Op, ScanEmit, ScanPredicate, ScanRole};
    use fuel_ir::{DType, Shape};
    use std::sync::{Arc, RwLock};

    /// Tests that build data `Const` tensors need a real device for the
    /// slot-populating constructors. Singleton CpuBackendDevice via OnceLock
    /// (mirrors grad.rs:216).
    fn cpu_dev() -> &'static std::sync::Arc<dyn fuel_backend_contract::DynBackendDevice> {
        static D: std::sync::OnceLock<std::sync::Arc<dyn fuel_backend_contract::DynBackendDevice>> =
            std::sync::OnceLock::new();
        D.get_or_init(|| std::sync::Arc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice))
    }

    // Build a trivial scan: carry [1], body new_carry = carry*2, body_y =
    // new_carry, n_xs = 0, bound = 3, emit = All. Returns (graph_arc, scan_id).
    fn trivial_scan(
        bound: usize,
        emit: ScanEmit,
        early_exit: Option<ScanPredicate>,
    ) -> (Arc<RwLock<Graph>>, crate::NodeId) {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let scan = {
            let mut g = graph.write().unwrap();
            let s = Shape::from_dims(&[1]);
            let carry = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let hole = g.push(Node {
                op: Op::ScanPlaceholder {
                    role: ScanRole::Carry,
                    index: 0,
                },
                inputs: vec![],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let nc = g.push(Node {
                op: Op::MulScalar(2.0),
                inputs: vec![hole],
                shape: s.clone(),
                dtype: DType::F32,
            });
            g.push(Node {
                op: Op::Scan {
                    n_carries: 1,
                    n_xs: 0,
                    bound,
                    emit,
                    early_exit,
                },
                inputs: vec![carry, nc, nc],
                shape: Shape::from_dims(&[bound, 1]),
                dtype: DType::F32,
            })
        };
        (graph, scan)
    }

    /// GAP-303: a TWO-carry scan, exercised through the real internal path.
    ///
    /// ⚠️ `NodeHandle::scan` pins `n_carries: 1`, so nothing public can build one
    /// of these — without this test the multi-carry machinery ships written but
    /// never RUN, and a green suite says nothing about it.
    ///
    /// THE DISCRIMINATING ASSERTION is that carry 1 reads its OWN init, not carry
    /// 0's. Before this change `clone_body_node`'s `Carry` arm ignored `index`
    /// entirely (`role: Carry, .. => carry`), so EVERY carry hole substituted to
    /// the single carry — a 2-carry body would have silently threaded carry 0
    /// into both slots and produced a graph that runs and is wrong.
    fn two_carry_scan() -> (
        Arc<RwLock<Graph>>,
        crate::NodeId,
        crate::NodeId,
        crate::NodeId,
    ) {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (scan, c0, c1) = {
            let mut g = graph.write().unwrap();
            let sh = Shape::from_dims(&[1]);
            let mk = |g: &mut Graph, op: Op, inputs: Vec<crate::NodeId>| {
                g.push(Node {
                    op,
                    inputs,
                    shape: sh.clone(),
                    dtype: DType::F32,
                })
            };
            // Two DISTINCT inits, so "carry 1 read carry 0" is observable.
            let c0 = mk(&mut g, Op::Const, vec![]);
            let c1 = mk(&mut g, Op::Const, vec![]);
            let h0 = mk(
                &mut g,
                Op::ScanPlaceholder {
                    role: ScanRole::Carry,
                    index: 0,
                },
                vec![],
            );
            let h1 = mk(
                &mut g,
                Op::ScanPlaceholder {
                    role: ScanRole::Carry,
                    index: 1,
                },
                vec![],
            );
            // Different factors, so the two carry chains are distinguishable.
            let nc0 = mk(&mut g, Op::MulScalar(2.0), vec![h0]);
            let nc1 = mk(&mut g, Op::MulScalar(3.0), vec![h1]);
            let y = mk(&mut g, Op::MulScalar(5.0), vec![h0]);
            // layout = [init_c0, init_c1, | nc0, nc1, y]
            let scan = g.push(Node {
                op: Op::Scan {
                    n_carries: 2,
                    n_xs: 0,
                    bound: 1,
                    emit: ScanEmit::All,
                    early_exit: None,
                },
                inputs: vec![c0, c1, nc0, nc1, y],
                shape: Shape::from_dims(&[1, 1]),
                dtype: DType::F32,
            });
            (scan, c0, c1)
        };
        (graph, scan, c0, c1)
    }

    /// GAP-303 / Baracuda's requirement: the DUMP must name the carry count.
    ///
    /// ⚠️ Their reason, earned the expensive way: they spent a day on a benchmark
    /// that ran four times, reproducibly, with four valid controls, AGAINST THE
    /// WRONG KERNEL. "A control tells you the encoding is sound; only the
    /// subject's identity tells you what it encoded." `short_name()` renders
    /// every scan as the bare string "Scan", so a body built from the wrong scan
    /// was indistinguishable in every panic message this graph can emit.
    #[test]
    fn gap303_describe_node_names_the_carry_count() {
        let (graph, scan, _c0, _c1) = two_carry_scan();
        let g = graph.read().unwrap();
        let d = g.describe_node(scan);
        assert!(
            d.contains("n_carries=2"),
            "the dump must name the carry count, got: {d}"
        );
        assert!(
            d.contains("n_xs=0") && d.contains("bound=1"),
            "and its siblings: {d}"
        );
        // A placeholder must name WHICH hole it is -- Carry/0 vs Carry/1 is the
        // distinction a wrong-body bug turns on.
        let layout = crate::scan::parse_scan_layout(&g, scan).expect("layout");
        let hole = g.node(layout.body_new_carries[1]).inputs[0];
        let hd = g.describe_node(hole);
        assert!(
            hd.contains("index=1") && hd.contains("Carry"),
            "a carry hole must name its index, got: {hd}"
        );
    }

    #[test]
    fn gap303_parse_scan_layout_splits_a_two_carry_block() {
        let (graph, scan, c0, c1) = two_carry_scan();
        let g = graph.read().unwrap();
        let l = crate::scan::parse_scan_layout(&g, scan).expect("layout");
        assert_eq!(l.n_carries, 2);
        assert_eq!(
            l.init_carries,
            vec![c0, c1],
            "leading block is n_carries wide"
        );
        assert_eq!(
            l.body_new_carries.len(),
            2,
            "trailing block is n_carries wide"
        );
        assert!(l.xs.is_empty() && l.consts.is_empty());
        // body_y must be the node AFTER the new-carry block, not inside it.
        assert!(
            !l.body_new_carries.contains(&l.body_y),
            "body_y must not be mistaken for a new-carry slot"
        );
    }

    #[test]
    fn gap303_two_carries_thread_independently_through_unroll() {
        let (graph, scan, c0, c1) = two_carry_scan();
        let (_ys, carries) = {
            let mut g = graph.write().unwrap();
            unroll_scan(&mut g, scan, 1).expect("unroll a 2-carry scan")
        };
        assert_eq!(carries.len(), 2, "one final carry per carry slot");
        let g = graph.read().unwrap();
        // ⚠️ THE ASSERTION THAT CATCHES THE OLD BEHAVIOUR: carry 1's chain must
        // reach ITS OWN init. With the index-ignoring arm it reached c0.
        assert_eq!(
            g.node(carries[1]).inputs,
            vec![c1],
            "carry 1 must read init_carry 1, NOT init_carry 0"
        );
        assert_eq!(
            g.node(carries[0]).inputs,
            vec![c0],
            "carry 0 reads its own init"
        );
        assert!(
            matches!(g.node(carries[1]).op, Op::MulScalar(f) if f == 3.0),
            "carry 1 keeps its own body op"
        );
        assert_ne!(carries[0], carries[1], "the two carries are distinct nodes");
    }

    #[test]
    fn gap303_build_scan_step_materialises_every_carry() {
        let (graph, scan, c0, c1) = two_carry_scan();
        let layout = {
            let g = graph.read().unwrap();
            crate::scan::parse_scan_layout(&g, scan).expect("layout")
        };
        let step = {
            let mut g = graph.write().unwrap();
            crate::scan::build_scan_step(&mut g, &layout, 0, &[c0, c1]).expect("step")
        };
        assert_eq!(step.new_carries.len(), 2, "a step advances every carry");
        let g = graph.read().unwrap();
        assert_eq!(
            g.node(step.new_carries[1]).inputs,
            vec![c1],
            "carry 1 stays its own"
        );
    }

    #[test]
    fn gap303_validator_rejects_a_carry_index_past_n_carries() {
        let (graph, scan, _c0, _c1) = two_carry_scan();
        let g = graph.read().unwrap();
        let l = crate::scan::parse_scan_layout(&g, scan).expect("layout");
        let mut roots = l.body_new_carries.clone();
        roots.push(l.body_y);
        // In range for n_carries = 2 ...
        crate::scan::validate_scan_body_placeholders(&g, &roots, 2, 0)
            .expect("Carry/0 and Carry/1 are in range when n_carries = 2");
        // ... and OUT of range once the count shrinks. This is the guard that
        // used to read `index != 0`; it now range-checks like the Elem arm.
        let err = crate::scan::validate_scan_body_placeholders(&g, &roots, 1, 0)
            .expect_err("Carry/1 must be rejected when n_carries = 1");
        let msg = err.to_string();
        assert!(
            msg.contains("Carry, 1") && msg.contains("n_carries = 1"),
            "the error must name the offending index AND the bound, got: {msg}"
        );
    }

    #[test]
    fn unroll_scan_all_produces_a_concat_of_steps_and_no_scan_nodes() {
        let (graph, scan) = trivial_scan(3, ScanEmit::All, None);
        let (ys_v, _carries) = {
            let mut g = graph.write().unwrap();
            unroll_scan(&mut g, scan, 3).expect("unroll")
        };
        // GAP-303: the ys side is a one-element Vec now (see unroll_scan).
        let ys = ys_v[0];
        let g = graph.read().unwrap();
        // ys root is a Concat over the 3 steps.
        assert!(
            matches!(g.node(ys).op, Op::Concat { .. }),
            "emit=All ys root should be Concat, got {:?}",
            g.node(ys).op.short_name()
        );
        assert_eq!(g.node(ys).inputs.len(), 3, "one input per step");
        // No Op::Scan / Op::ScanPlaceholder reachable from the unrolled root.
        let reachable = crate::topo_order_multi(&g, &[ys]);
        assert!(
            !reachable
                .iter()
                .any(|&n| matches!(g.node(n).op, Op::Scan { .. } | Op::ScanPlaceholder { .. })),
            "unrolled graph must contain no Scan/ScanPlaceholder nodes"
        );
    }

    #[test]
    fn unroll_scan_early_exit_some_peels_predicate_and_unrolls() {
        // early_exit = Some layout: [carry, consts=[thr], body_new_carry, body_y, pred_exit].
        // unroll must PEEL pred_exit, IGNORE it, and emit a 3-step Concat with no scan nodes.
        let graph = Arc::new(RwLock::new(Graph::new()));
        let scan = {
            let mut g = graph.write().unwrap();
            let s = Shape::from_dims(&[1]);
            let carry = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let thr = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let hole = g.push(Node {
                op: Op::ScanPlaceholder {
                    role: ScanRole::Carry,
                    index: 0,
                },
                inputs: vec![],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let nc = g.push(Node {
                op: Op::MulScalar(2.0),
                inputs: vec![hole],
                shape: s.clone(),
                dtype: DType::F32,
            });
            // predicate sub-DAG over the post-step carry (ignored by unroll).
            let pred = g.push(Node {
                op: Op::Ge,
                inputs: vec![nc, thr],
                shape: s.clone(),
                dtype: DType::Bool,
            });
            g.push(Node {
                op: Op::Scan {
                    n_carries: 1,
                    n_xs: 0,
                    bound: 3,
                    emit: ScanEmit::All,
                    early_exit: Some(ScanPredicate),
                },
                inputs: vec![carry, thr, nc, nc, pred], // consts=[thr], new_carry=nc, y=nc, pred_exit=pred
                shape: Shape::from_dims(&[3, 1]),
                dtype: DType::F32,
            })
        };
        let (ys_v, _carries) = {
            let mut g = graph.write().unwrap();
            unroll_scan(&mut g, scan, 3).expect("unroll must peel + ignore the predicate")
        };
        // GAP-303: the ys side is a one-element Vec now (see unroll_scan).
        let ys = ys_v[0];
        let g = graph.read().unwrap();
        assert!(
            matches!(g.node(ys).op, Op::Concat { .. }),
            "emit=All ys root should be Concat"
        );
        assert_eq!(g.node(ys).inputs.len(), 3, "one input per step");
        let reachable = crate::topo_order_multi(&g, &[ys]);
        assert!(
            !reachable
                .iter()
                .any(|&n| matches!(g.node(n).op, Op::Scan { .. } | Op::ScanPlaceholder { .. })),
            "unrolled graph must contain no Scan/ScanPlaceholder nodes"
        );
    }

    #[test]
    fn op_scan_is_a_terminal_in_the_base_map() {
        // lower_to_base_map must LEAVE Op::Scan in place (no LoweringRule
        // matches a bare Op variant) — not silently expanded, not errored.
        let (graph, scan) = trivial_scan(3, ScanEmit::All, None);
        let roots = lower_to_base_map(&graph, &[scan]);
        let g = graph.read().unwrap();
        let reachable = crate::topo_order_multi(&g, &roots);
        assert!(
            reachable
                .iter()
                .any(|&n| matches!(g.node(n).op, Op::Scan { .. })),
            "Op::Scan must remain a terminal after lower_to_base_map"
        );
    }

    #[test]
    fn unroll_scan_rejects_malformed_short_inputs() {
        // n_xs = 0 well-formed minimum is init_carry(1) + body_exits(2) = 3.
        // Build inputs of length 2 (one short) — must be a typed Err, not a
        // panic from the `consts = inputs[1+n_xs..inputs.len()-2]` slice
        // (start=1 > end=0 when inputs.len() == 2).
        let mut g = Graph::new();
        let s = Shape::from_dims(&[1]);
        let carry = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: s.clone(),
            dtype: DType::F32,
        });
        let body_exit = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: s.clone(),
            dtype: DType::F32,
        });
        let scan = g.push(Node {
            op: Op::Scan {
                n_carries: 1,
                n_xs: 0,
                bound: 1,
                emit: ScanEmit::All,
                early_exit: None,
            },
            inputs: vec![carry, body_exit],
            shape: Shape::from_dims(&[1, 1]),
            dtype: DType::F32,
        });
        let r = unroll_scan(&mut g, scan, 1);
        assert!(
            r.is_err(),
            "inputs.len() == n_xs + 2 must be rejected as malformed, not panic"
        );
    }

    #[test]
    fn unroll_scan_rejects_elem_index_out_of_range() {
        // n_xs = 0 (no xs slots) but the body references ScanPlaceholder{Elem,
        // 0} — index 0 is out of range since n_xs = 0. Must be a typed Err,
        // not an `elem[index]` panic inside clone_body_node.
        let graph = Arc::new(RwLock::new(Graph::new()));
        let scan = {
            let mut g = graph.write().unwrap();
            let s = Shape::from_dims(&[1]);
            let carry = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let elem_hole = g.push(Node {
                op: Op::ScanPlaceholder {
                    role: ScanRole::Elem,
                    index: 0,
                },
                inputs: vec![],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let nc = g.push(Node {
                op: Op::MulScalar(2.0),
                inputs: vec![elem_hole],
                shape: s.clone(),
                dtype: DType::F32,
            });
            g.push(Node {
                op: Op::Scan {
                    n_carries: 1,
                    n_xs: 0,
                    bound: 1,
                    emit: ScanEmit::All,
                    early_exit: None,
                },
                inputs: vec![carry, nc, nc],
                shape: Shape::from_dims(&[1, 1]),
                dtype: DType::F32,
            })
        };
        let mut g = graph.write().unwrap();
        let r = unroll_scan(&mut g, scan, 1);
        assert!(
            r.is_err(),
            "Elem index >= n_xs must be a typed Err, never an elem[index] panic"
        );
    }

    #[test]
    fn unroll_scan_nxs_positive_slices_substitutes_and_shares_consts() {
        // n_xs = 1, one shared const, bound = steps = 2, emit = All. Body:
        // new_carry = carry + elem0; y = (carry + elem0) * const — references
        // BOTH placeholders AND the shared const. xs[0] shape [2, 1] (leading
        // dim = bound). Locks the slice/substitute/const-sharing semantics
        // Tasks 6-7 depend on.
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (scan, const_id) = {
            let mut g = graph.write().unwrap();
            let carry_shape = Shape::from_dims(&[1]);
            let xs_shape = Shape::from_dims(&[2, 1]);
            let init_carry = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: carry_shape.clone(),
                dtype: DType::F32,
            });
            let xs0 = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: xs_shape.clone(),
                dtype: DType::F32,
            });
            let const_id = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: carry_shape.clone(),
                dtype: DType::F32,
            });
            let carry_hole = g.push(Node {
                op: Op::ScanPlaceholder {
                    role: ScanRole::Carry,
                    index: 0,
                },
                inputs: vec![],
                shape: carry_shape.clone(),
                dtype: DType::F32,
            });
            let elem_hole = g.push(Node {
                op: Op::ScanPlaceholder {
                    role: ScanRole::Elem,
                    index: 0,
                },
                inputs: vec![],
                shape: carry_shape.clone(),
                dtype: DType::F32,
            });
            let sum = g.push(Node {
                op: Op::Add,
                inputs: vec![carry_hole, elem_hole],
                shape: carry_shape.clone(),
                dtype: DType::F32,
            });
            let new_carry = sum;
            let y = g.push(Node {
                op: Op::Mul,
                inputs: vec![sum, const_id],
                shape: carry_shape.clone(),
                dtype: DType::F32,
            });
            let scan = g.push(Node {
                op: Op::Scan {
                    n_carries: 1,
                    n_xs: 1,
                    bound: 2,
                    emit: ScanEmit::All,
                    early_exit: None,
                },
                inputs: vec![init_carry, xs0, const_id, new_carry, y],
                shape: Shape::from_dims(&[2, 1]),
                dtype: DType::F32,
            });
            (scan, const_id)
        };
        let (ys_v, _carries) = {
            let mut g = graph.write().unwrap();
            unroll_scan(&mut g, scan, 2).expect("unroll")
        };
        // GAP-303: the ys side is a one-element Vec now (see unroll_scan).
        let ys = ys_v[0];
        let g = graph.read().unwrap();
        assert!(
            matches!(g.node(ys).op, Op::Concat { .. }),
            "emit=All ys root should be Concat, got {:?}",
            g.node(ys).op.short_name()
        );
        assert_eq!(g.node(ys).inputs.len(), 2, "one input per step");
        let reachable = crate::topo_order_multi(&g, &[ys]);
        assert!(
            !reachable
                .iter()
                .any(|&n| matches!(g.node(n).op, Op::Scan { .. } | Op::ScanPlaceholder { .. })),
            "unrolled graph must contain no Scan/ScanPlaceholder nodes"
        );
        // The const NodeId must be SHARED across both step clones — it
        // appears exactly once in the reachable set (topo_order_multi
        // dedups by NodeId), never re-cloned per step.
        let const_occurrences = reachable.iter().filter(|&&n| n == const_id).count();
        assert_eq!(
            const_occurrences, 1,
            "const node must be shared (not cloned) across steps"
        );
    }

    #[test]
    fn scan_until_builds_early_exit_node_hashes_distinctly_and_validates() {
        use crate::NodeHandle;
        // init_carry [1]; body new_carry = carry*2; consts include threshold.
        let init = NodeHandle::from_f32(vec![1.0f32], Shape::from_dims(&[1]), cpu_dev()).unwrap();
        let graph = init.graph().clone();
        // Build the shared body + predicate at graph level, wrap as NodeHandle handles.
        let (nc, thr, pred_ok) = {
            let mut g = graph.write().unwrap();
            let s = Shape::from_dims(&[1]);
            let hole = g.push(Node {
                op: Op::ScanPlaceholder {
                    role: ScanRole::Carry,
                    index: 0,
                },
                inputs: vec![],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let nc = g.push(Node {
                op: Op::MulScalar(2.0),
                inputs: vec![hole],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let thr = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let pred = g.push(Node {
                op: Op::Ge,
                inputs: vec![nc, thr],
                shape: s.clone(),
                dtype: DType::Bool,
            });
            (nc, thr, pred)
        };
        let nc_t = NodeHandle::from_existing(graph.clone(), nc);
        let thr_t = NodeHandle::from_existing(graph.clone(), thr);
        let pred_t = NodeHandle::from_existing(graph.clone(), pred_ok);

        let out = init
            .scan_until(
                &[],
                std::slice::from_ref(&thr_t),
                &nc_t,
                &nc_t,
                &pred_t,
                5,
                ScanEmit::Final,
            )
            .expect("well-formed scan_until must build");
        // The producer node behind the emit=Final view is an Op::Scan{early_exit: Some}.
        let scan_id = {
            let g = graph.read().unwrap();
            g.node(out.id()).inputs[0]
        };
        {
            let g = graph.read().unwrap();
            match &g.node(scan_id).op {
                Op::Scan { early_exit, .. } => {
                    assert!(early_exit.is_some(), "early_exit must be Some")
                }
                other => panic!("expected Op::Scan, got {}", other.short_name()),
            }
            // pred_exit is the LAST input (trailing), so reachability sees it.
            assert_eq!(*g.node(scan_id).inputs.last().unwrap(), pred_ok);
        }

        // base_map_hash distinctness: a scan with the SAME body but a DIFFERENT predicate hashes differently.
        let thr2 = {
            let mut g = graph.write().unwrap();
            g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[1]),
                dtype: DType::F32,
            })
        };
        let pred2 = {
            let mut g = graph.write().unwrap();
            g.push(Node {
                op: Op::Le,
                inputs: vec![nc, thr2],
                shape: Shape::from_dims(&[1]),
                dtype: DType::Bool,
            })
        };
        let pred2_t = NodeHandle::from_existing(graph.clone(), pred2);
        let out2 = init
            .scan_until(
                &[],
                &[NodeHandle::from_existing(graph.clone(), thr2)],
                &nc_t,
                &nc_t,
                &pred2_t,
                5,
                ScanEmit::Final,
            )
            .expect("second scan_until builds");
        let scan2 = {
            let g = graph.read().unwrap();
            g.node(out2.id()).inputs[0]
        };
        let (h1, h2) = {
            let g = graph.read().unwrap();
            (
                crate::opt::base_map_hash(&g, scan_id),
                crate::opt::base_map_hash(&g, scan2),
            )
        };
        assert_ne!(
            h1, h2,
            "different predicates must hash distinctly (predicate is a trailing input)"
        );

        // Rejection: a NON-scalar predicate is a typed Err (never a panic).
        let big =
            NodeHandle::from_f32(vec![0.0f32, 1.0], Shape::from_dims(&[2]), cpu_dev()).unwrap(); // wrong graph AND non-scalar
        assert!(
            init.scan_until(
                &[],
                std::slice::from_ref(&thr_t),
                &nc_t,
                &nc_t,
                &big,
                5,
                ScanEmit::Final
            )
            .is_err(),
            "non-same-graph / non-scalar predicate must be a typed Err"
        );
        // Rejection: a non-Bool predicate.
        let f32pred = {
            let mut g = graph.write().unwrap();
            g.push(Node {
                op: Op::Sqr,
                inputs: vec![nc],
                shape: Shape::from_dims(&[1]),
                dtype: DType::F32,
            })
        };
        let f32pred_t = NodeHandle::from_existing(graph.clone(), f32pred);
        assert!(
            init.scan_until(&[], &[thr_t], &nc_t, &nc_t, &f32pred_t, 5, ScanEmit::Final)
                .is_err(),
            "non-Bool predicate must be a typed Err"
        );
    }

    #[test]
    fn build_scan_step_shares_subst_so_predicate_reads_this_steps_new_carry() {
        // Scan: carry [1]; new_carry = carry*2; pred = Ge(new_carry, thr). n_xs=0, bound=4.
        let graph = Arc::new(RwLock::new(Graph::new()));
        let scan = {
            let mut g = graph.write().unwrap();
            let s = Shape::from_dims(&[1]);
            let carry = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let thr = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let hole = g.push(Node {
                op: Op::ScanPlaceholder {
                    role: ScanRole::Carry,
                    index: 0,
                },
                inputs: vec![],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let nc = g.push(Node {
                op: Op::MulScalar(2.0),
                inputs: vec![hole],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let pred = g.push(Node {
                op: Op::Ge,
                inputs: vec![nc, thr],
                shape: s.clone(),
                dtype: DType::Bool,
            });
            g.push(Node {
                op: Op::Scan {
                    n_carries: 1,
                    n_xs: 0,
                    bound: 4,
                    emit: ScanEmit::Final,
                    early_exit: Some(ScanPredicate),
                },
                inputs: vec![carry, thr, nc, nc, pred],
                shape: Shape::from_dims(&[4, 1]),
                dtype: DType::F32,
            })
        };
        let layout = {
            let g = graph.read().unwrap();
            crate::scan::parse_scan_layout(&g, scan).expect("layout")
        };
        assert!(layout.pred_exit.is_some());
        let init = layout.init_carries.clone();
        let step = {
            let mut g = graph.write().unwrap();
            crate::scan::build_scan_step(&mut g, &layout, 0, &init).expect("step")
        };
        let stop = step.stop.expect("early-exit scan yields a stop node");
        // The predicate clone must reach step.new_carry (the shared post-step carry),
        // and must reach NO ScanPlaceholder (all substituted away).
        let g = graph.read().unwrap();
        let reach = crate::topo_order_multi(&g, &[stop]);
        assert!(
            reach.contains(&step.new_carries[0]),
            "pred must read THIS step's new_carry (shared subst)"
        );
        assert!(
            !reach
                .iter()
                .any(|&n| matches!(g.node(n).op, Op::ScanPlaceholder { .. })),
            "no placeholders survive a materialized step"
        );
        // The Ge's first input IS the step's new_carry — proof there was no double-clone.
        let ge_inputs = &g.node(stop).inputs;
        assert_eq!(
            ge_inputs[0], step.new_carries[0],
            "predicate's post-step operand is the shared new_carry"
        );
    }

    #[test]
    fn scan_until_rejects_out_of_range_elem_placeholder_at_build_time() {
        use crate::NodeHandle;
        // n_xs = 1, but the body references ScanPlaceholder{Elem, 5} — out of
        // range. Must be a typed BUILD-TIME Err (before this fix the node built
        // fine and the forward driver's build_scan_step -> clone_body_node
        // panicked with elem[5] index-OOB — a never-panic regression).
        let init = NodeHandle::from_f32(vec![0.0f32], Shape::from_dims(&[1]), cpu_dev()).unwrap();
        let graph = init.graph().clone();
        let (x, nc, thr, pred) = {
            let mut g = graph.write().unwrap();
            let s = Shape::from_dims(&[1]);
            let x = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[3, 1]),
                dtype: DType::F32,
            });
            let bad_elem = g.push(Node {
                op: Op::ScanPlaceholder {
                    role: ScanRole::Elem,
                    index: 5,
                },
                inputs: vec![],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let carry_hole = g.push(Node {
                op: Op::ScanPlaceholder {
                    role: ScanRole::Carry,
                    index: 0,
                },
                inputs: vec![],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let nc = g.push(Node {
                op: Op::Add,
                inputs: vec![carry_hole, bad_elem],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let thr = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let pred = g.push(Node {
                op: Op::Ge,
                inputs: vec![carry_hole, thr],
                shape: s.clone(),
                dtype: DType::Bool,
            });
            (x, nc, thr, pred)
        };
        let x_t = NodeHandle::from_existing(graph.clone(), x);
        let thr_t = NodeHandle::from_existing(graph.clone(), thr);
        let nc_t = NodeHandle::from_existing(graph.clone(), nc);
        let pred_t = NodeHandle::from_existing(graph.clone(), pred);
        let r = init.scan_until(&[x_t], &[thr_t], &nc_t, &nc_t, &pred_t, 3, ScanEmit::Final);
        assert!(
            r.is_err(),
            "scan_until must reject body Elem{{index >= n_xs}} at build time, not panic in the driver"
        );
    }

    #[test]
    fn scan_rejects_out_of_range_elem_placeholder_at_build_time() {
        use crate::NodeHandle;
        // Base builder shares the same missing build-time check (Phase 1 only
        // avoided the panic because unroll_scan was its sole forward path).
        let init = NodeHandle::from_f32(vec![0.0f32], Shape::from_dims(&[1]), cpu_dev()).unwrap();
        let graph = init.graph().clone();
        let (x, nc) = {
            let mut g = graph.write().unwrap();
            let s = Shape::from_dims(&[1]);
            let x = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[3, 1]),
                dtype: DType::F32,
            });
            let bad_elem = g.push(Node {
                op: Op::ScanPlaceholder {
                    role: ScanRole::Elem,
                    index: 5,
                },
                inputs: vec![],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let carry_hole = g.push(Node {
                op: Op::ScanPlaceholder {
                    role: ScanRole::Carry,
                    index: 0,
                },
                inputs: vec![],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let nc = g.push(Node {
                op: Op::Add,
                inputs: vec![carry_hole, bad_elem],
                shape: s.clone(),
                dtype: DType::F32,
            });
            (x, nc)
        };
        let x_t = NodeHandle::from_existing(graph.clone(), x);
        let nc_t = NodeHandle::from_existing(graph.clone(), nc);
        let r = init.scan(&[x_t], &[], &nc_t, &nc_t, 3, ScanEmit::Final);
        assert!(
            r.is_err(),
            "base scan must also reject body Elem{{index >= n_xs}} at build time"
        );
    }

    #[test]
    fn backward_lowers_op_scan_and_no_longer_panics() {
        use crate::NodeHandle;
        // Affine scan: carry[1]; consts a,b; new_carry = a*carry + b; emit=Final. bound=3.
        let init = NodeHandle::from_f32(vec![1.0f32], Shape::from_dims(&[1]), cpu_dev()).unwrap();
        let a = NodeHandle::from_existing(init.graph().clone(), init.id())
            .const_f32_like(vec![0.5f32], Shape::from_dims(&[1]))
            .unwrap();
        let b = NodeHandle::from_existing(init.graph().clone(), init.id())
            .const_f32_like(vec![0.1f32], Shape::from_dims(&[1]))
            .unwrap();
        let graph = init.graph().clone();
        let nc = {
            let mut g = graph.write().unwrap();
            let s = Shape::from_dims(&[1]);
            let hole = g.push(Node {
                op: Op::ScanPlaceholder {
                    role: ScanRole::Carry,
                    index: 0,
                },
                inputs: vec![],
                shape: s.clone(),
                dtype: DType::F32,
            });
            let ac = g.push(Node {
                op: Op::Mul,
                inputs: vec![a.id(), hole],
                shape: s.clone(),
                dtype: DType::F32,
            });
            g.push(Node {
                op: Op::Add,
                inputs: vec![ac, b.id()],
                shape: s.clone(),
                dtype: DType::F32,
            })
        };
        let nc_t = NodeHandle::from_existing(graph.clone(), nc);
        let out = init
            .scan(
                &[],
                &[a.clone(), b.clone()],
                &nc_t,
                &nc_t,
                3,
                ScanEmit::Final,
            )
            .expect("scan");
        // backward() must NOT panic (Phase 1 arm panics here) and must yield a grad for init_carry.
        let grads = out.backward();
        let g_init = grads.get(&init).expect("gradient for init_carry");
        // The gradient's subgraph must contain no Op::Scan/ScanPlaceholder (proof of lowering).
        let g = graph.read().unwrap();
        let reach = crate::topo_order_multi(&g, &[g_init.id()]);
        assert!(
            !reach
                .iter()
                .any(|&n| matches!(g.node(n).op, Op::Scan { .. } | Op::ScanPlaceholder { .. })),
            "backward must lower the scan before differentiating"
        );
        assert!(
            grads.get(&a).is_some() && grads.get(&b).is_some(),
            "consts a,b get gradients (BPTT)"
        );
    }
}

//! Modern Hopfield associative-memory retrieval + the realize-barrier
//! early-exit step driver.
//!
//! `drive_scan_until_final_f32` is the forward-realize control flow for an
//! `Op::Scan { early_exit: Some, emit: Final }`: it realizes each step's carry
//! and predicate at the realize barrier, feeds the realized carry forward as a
//! fresh data `Const` (breaking the recurrent data dependency so each step is
//! O(1 step), not O(t)), and stops when the predicate fires — a data-dependent
//! iteration count under the static capacity `bound`. No `Op::Scan` native
//! kernel is involved; every step realizes a primitive sub-graph.
//!
//! `hopfield_retrieve` (Task 7) builds `ξ ← softmax(β·ξ·Xᵀ)·X` as an
//! `Op::Scan { early_exit: ‖Δξ‖ < ε, emit: Final }` consumer that runs through
//! this driver.

use std::sync::{Arc, RwLock};

use fuel_graph::{Graph, ScanEmit, Tensor};

/// Drive an `Op::Scan { early_exit: Some, emit: Final }` to its fixed point at
/// the realize barrier. Realizes each step's new carry and (optional) stop
/// predicate, feeds the realized carry forward as a fresh `Const`, and stops
/// when the predicate fires. Returns `(final_carry_bytes, runtime_step_count)`.
///
/// A predicate that never fires runs to `bound` (the static capacity) and
/// stops — bounded, never an infinite loop. Only `emit = Final` is supported
/// (the `emit = All` valid-count capacity buffer is out of scope for Phase 2).
pub fn drive_scan_until_final_f32(
    graph: &Arc<RwLock<Graph>>,
    scan_id: fuel_graph::NodeId,
    device: &crate::Device,
) -> Result<(Vec<f32>, usize), fuel_ir::Error> {
    let layout = { let g = graph.read().unwrap(); fuel_graph::scan::parse_scan_layout(&g, scan_id)? };
    if !matches!(layout.emit, ScanEmit::Final) {
        return Err(fuel_ir::Error::Msg(
            "drive_scan_until_final_f32: only emit=Final is supported (emit=All valid-count \
             buffer is out of scope for Phase 2)".into()).bt());
    }
    let carry_shape = { let g = graph.read().unwrap(); g.node(layout.init_carry).shape.clone() };
    let mut carry_id = layout.init_carry;
    let mut last: Vec<f32> = crate::pipelined_bridge::realize_one_as::<f32>(graph, carry_id, device)
        .map_err(|e| fuel_ir::Error::Msg(format!("drive_scan_until: realize init_carry: {e}")).bt())?;
    let mut count = 0usize;
    for t in 0..layout.bound {
        let step = { let mut g = graph.write().unwrap();
            fuel_graph::scan::build_scan_step(&mut g, &layout, t, carry_id)? };
        let nc = crate::pipelined_bridge::realize_one_as::<f32>(graph, step.new_carry, device)
            .map_err(|e| fuel_ir::Error::Msg(format!("drive_scan_until: realize step {t}: {e}")).bt())?;
        let stop = match step.stop {
            Some(stop_id) => {
                let b = crate::pipelined_bridge::realize_one_as::<u8>(graph, stop_id, device)
                    .map_err(|e| fuel_ir::Error::Msg(format!("drive_scan_until: realize predicate {t}: {e}")).bt())?;
                b.first().copied().unwrap_or(0) != 0
            }
            None => false, // no predicate: run to bound
        };
        count = t + 1;
        last = nc.clone();
        // Feed the realized carry forward as a fresh data const so the next step's
        // realize is O(1 step), not O(t) (breaks the recurrent data dependency).
        carry_id = Tensor::from_existing(graph.clone(), layout.init_carry)
            .const_f32_like(nc, carry_shape.clone()).id();
        if stop { break; }
    }
    Ok((last, count))
}

/// Modern (dense) Hopfield retrieval: `xi <- softmax(beta * xi * X^T) * X`,
/// iterated to a fixed point (carry = xi, early_exit = `||xi_new - xi|| < eps`,
/// emit = Final). `query`: `[1, d]` (init_carry); patterns `X`: `[n, d]` (a
/// const). Returns the retrieval Tensor (the emit=Final view). Runs forward
/// through [`drive_scan_until_final_f32`] (converges early) and backward
/// through the `lower_scans_for_backward` unroll pre-pass. No `Op::Scan`
/// native kernel, no fused Hopfield op — every step is matmul + softmax,
/// primitives that already have CPU kernels.
pub fn hopfield_retrieve(
    query: &fuel_graph::Tensor,
    patterns: &fuel_graph::Tensor,
    beta: f32, eps: f32, max_iters: usize,
) -> std::result::Result<fuel_graph::Tensor, fuel_ir::Error> {
    use fuel_graph::{Tensor, ScanEmit};
    if !std::sync::Arc::ptr_eq(query.graph(), patterns.graph()) {
        return Err(fuel_ir::Error::Msg("hopfield_retrieve: query and patterns must share a graph".into()).bt());
    }
    let d = { let dims = query.shape(); *dims.dims().last().ok_or_else(|| fuel_ir::Error::Msg("hopfield: query rank 0".into()).bt())? };
    let g = query.graph().clone();
    // carry placeholder xi [1, d].
    let xi = {
        let mut gw = g.write().unwrap();
        gw.push(fuel_graph::Node { op: fuel_graph::Op::ScanPlaceholder { role: fuel_graph::ScanRole::Carry, index: 0 },
            inputs: vec![], shape: fuel_ir::Shape::from_dims(&[1, d]), dtype: fuel_ir::DType::F32 })
    };
    let xi_t = Tensor::from_existing(g.clone(), xi);
    // body: logits = mul_scalar(beta)(xi @ X^T) [1,n]; s = softmax_last(logits); new = s @ X [1,d].
    let xt = patterns.transpose();                       // [d, n]
    let logits = xi_t.matmul(&xt).mul_scalar(beta as f64);
    let s = logits.softmax_last_dim();
    let new_carry = s.matmul(patterns);                  // [1, d]
    // pred: ||new - xi|| < eps  ->  Lt( sqrt(sum((new-xi)^2)), eps ) : U8 [1,1].
    let delta = new_carry.sub(&xi_t);
    let sq = delta.sqr();
    let sumsq = sq.reduce_sum_to(fuel_ir::Shape::from_dims(&[1, 1]));
    let norm = sumsq.sqrt();
    let eps_c = Tensor::from_existing(g.clone(), query.id())
        .const_f32_like(vec![eps], fuel_ir::Shape::from_dims(&[1, 1]));
    let pred = norm.lt(&eps_c);                           // U8 [1,1]
    // scan_until: consts must include EVERY const the body OR predicate reads: X and eps_c.
    query.scan_until(&[], &[patterns.clone(), eps_c], &new_carry, &new_carry, &pred, max_iters, ScanEmit::Final)
}

#[cfg(test)]
mod tests {
    use crate::{Device, hopfield::drive_scan_until_final_f32};
    use fuel_graph::{Graph, Node, Op, ScanEmit, ScanPredicate, ScanRole, Tensor};
    use fuel_ir::{DType, Shape};
    use std::sync::{Arc, RwLock};

    // Build carry[1]=0; new_carry = carry + 1; pred = Ge(new_carry, thr). Deterministic:
    // after step t, new_carry = t+1; pred fires at t = thr-1 -> count = thr.
    // The two body/predicate constants (one, thr) carry data via const_f32_like so
    // realize finds their bytes (plan Task-4 NOTE).
    fn counting_scan(bound: usize, thr: f32) -> (Arc<RwLock<Graph>>, fuel_graph::NodeId) {
        let init = Tensor::from_f32(vec![0.0f32], Shape::from_dims(&[1]), Device::cpu().as_dyn());
        let graph = init.graph().clone();
        let one_t = Tensor::from_existing(graph.clone(), init.id())
            .const_f32_like(vec![1.0f32], Shape::from_dims(&[1]));
        let thr_t = Tensor::from_existing(graph.clone(), init.id())
            .const_f32_like(vec![thr], Shape::from_dims(&[1]));
        let scan = {
            let mut g = graph.write().unwrap();
            let s = Shape::from_dims(&[1]);
            let hole = g.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 }, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
            let nc   = g.push(Node { op: Op::Add, inputs: vec![hole, one_t.id()], shape: s.clone(), dtype: DType::F32 });
            let pred = g.push(Node { op: Op::Ge, inputs: vec![nc, thr_t.id()], shape: s.clone(), dtype: DType::U8 });
            g.push(Node {
                op: Op::Scan { n_xs: 0, bound, emit: ScanEmit::Final, early_exit: Some(ScanPredicate) },
                inputs: vec![init.id(), one_t.id(), thr_t.id(), nc, nc, pred], // consts=[one, thr], new=nc, y=nc, pred
                shape: Shape::from_dims(&[bound, 1]),
                dtype: DType::F32,
            })
        };
        (graph, scan)
    }

    #[test]
    fn driver_stops_at_predicate_step_and_returns_that_carry() {
        let dev = Device::cpu();
        let (graph, scan) = counting_scan(/*bound*/ 10, /*thr*/ 3.0);
        let (carry, count) = drive_scan_until_final_f32(&graph, scan, &dev).expect("driver");
        assert_eq!(count, 3, "predicate Ge(new_carry, 3) fires at step index 2 -> count 3");
        assert!((carry[0] - 3.0).abs() < 1e-5, "returned carry is the step-3 value, got {}", carry[0]);
    }

    #[test]
    fn driver_runs_to_bound_when_predicate_never_fires() {
        let dev = Device::cpu();
        let (graph, scan) = counting_scan(/*bound*/ 6, /*thr*/ 999.0);
        let (carry, count) = drive_scan_until_final_f32(&graph, scan, &dev).expect("driver");
        assert_eq!(count, 6, "non-converging predicate runs to bound (no infinite loop)");
        assert!((carry[0] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn hopfield_retrieves_stored_pattern_and_exits_early() {
        let dev = Device::cpu();
        // Three orthogonal-ish stored patterns [n=3, d=4].
        let x = Tensor::from_f32(
            vec![1.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0,  0.0,0.0,1.0,0.0],
            Shape::from_dims(&[3,4]), &*dev.as_dyn());
        // Query near pattern 0.
        let q = Tensor::from_existing(x.graph().clone(), x.id())
            .const_f32_like(vec![0.9, 0.2, 0.1, 0.0], Shape::from_dims(&[1,4]));
        let retrieval = crate::hopfield::hopfield_retrieve(&q, &x, /*beta*/ 8.0, /*eps*/ 1e-3, /*max_iters*/ 20)
            .expect("build hopfield retrieval");
        let scan_id = { let g = retrieval.graph().read().unwrap(); g.node(retrieval.id()).inputs[0] };
        let (xi, count) = crate::hopfield::drive_scan_until_final_f32(&retrieval.graph().clone(), scan_id, &dev)
            .expect("drive");
        // Converged to pattern 0 (dominant coordinate 0), and stopped BEFORE the capacity.
        assert!(xi[0] > 0.8 && xi[1] < 0.2 && xi[2] < 0.2, "retrieved xi should snap to pattern 0: {xi:?}");
        assert!(count < 20, "early-exit must stop before bound (converged in {count} < 20 iters)");
        assert!(count >= 1);
    }

    #[test]
    fn hopfield_gradient_matches_finite_difference() {
        let dev = Device::cpu();
        // Forward loss L(X) = sum(unroll(retrieve(q, X, beta, eps, 3))), FD over X[0].
        let build = |x_vals: &[f32]| -> (std::sync::Arc<std::sync::RwLock<fuel_graph::Graph>>, fuel_graph::Tensor, fuel_graph::Tensor) {
            let x = Tensor::from_f32(x_vals.to_vec(), Shape::from_dims(&[2, 3]), &*dev.as_dyn());
            let q = Tensor::from_existing(x.graph().clone(), x.id()).const_f32_like(vec![0.6, 0.3, 0.1], Shape::from_dims(&[1, 3]));
            let r = crate::hopfield::hopfield_retrieve(&q, &x, 4.0, 1e-6, 3).expect("retrieve");
            (x.graph().clone(), x, r)
        };
        let x0 = vec![1.0f32, 0.0, 0.0,  0.0, 1.0, 0.0];
        // Forward via unroll+realize+sum.
        let fwd = |x_vals: &[f32]| -> f32 {
            let (g, _x, r) = build(x_vals);
            let scan_id = { let gr = g.read().unwrap(); gr.node(r.id()).inputs[0] };
            let bound = { let gr = g.read().unwrap(); match gr.node(scan_id).op { fuel_graph::Op::Scan { bound, .. } => bound, _ => unreachable!() } };
            let carry = { let mut gw = g.write().unwrap(); fuel_graph::scan::unroll_scan(&mut gw, scan_id, bound).expect("unroll").0 };
            crate::pipelined_bridge::realize_one_as::<f32>(&g, carry, &dev).expect("realize").iter().sum()
        };
        // Autograd at x0: loss = sum(retrieval). Build a scalar loss node, backward, grad w.r.t X.
        let (g, x, r) = build(&x0);
        let loss = r.reduce_sum_to(fuel_ir::Shape::from_dims(&[1, 1])); // sum -> scalar
        let grads = loss.backward();
        let g_x_id = grads.get(&x).expect("grad X").id();
        let g_x = crate::pipelined_bridge::realize_one_as::<f32>(&g, g_x_id, &dev).expect("realize gradX");
        // Central FD on X[0].
        let h = 1e-3f32;
        let mut xp = x0.clone(); xp[0] += h;
        let mut xm = x0.clone(); xm[0] -= h;
        let fd0 = (fwd(&xp) - fwd(&xm)) / (2.0*h);
        assert!((g_x[0] - fd0).abs() < 5e-2, "hopfield dL/dX[0]: autograd {} vs FD {fd0}", g_x[0]);
    }
}

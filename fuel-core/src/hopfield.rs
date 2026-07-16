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
}

//! End-to-end integration test for the **production runtime-fusion pipeline**
//! (`optimize_graph_with_runtime_fusion` → `RuntimeFusedArmPathfinder` →
//! `offer_runtime_fused_arm`): adopt a runtime op, optimize a graph containing
//! its region, and observe the gated `Op::Branch` fused arm in the optimized
//! graph — the "the prod constructor isn't the untested one" gate from the
//! dd-shapes coordination (2026-07-08).
//!
//! Lives in `tests/` (its own process) so the process-global runtime-fused
//! sidecar is hermetic by construction w.r.t. the lib unit tests. Everything
//! here runs in ONE `#[test]` fn — adopting tests in a shared process must
//! serialize, and one fn is the degenerate (free) serialization.

use std::collections::HashMap;
use std::sync::{Arc, RwLock as StdRwLock};

use fuel_dispatch::PlanOptions;
use fuel_dispatch::optimize::{optimize_graph, optimize_graph_with_runtime_fusion};
use fuel_dispatch::runtime_fused_kernels::{adopt_runtime_fused, clear_runtime_fused_for_tests};
use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
use fuel_graph::registry::FusedOpParams;
use fuel_graph::{Graph, Node, NodeId, Op};
use fuel_ir::probe::BackendId;
use fuel_ir::{DType, DeviceLocation, Layout, Result, Shape};

fn noop_kernel(
    _inputs: &[Arc<StdRwLock<fuel_memory::Storage>>],
    _outputs: &mut [Arc<StdRwLock<fuel_memory::Storage>>],
    _layouts: &[Layout],
    _params: &fuel_dispatch::kernel::OpParams,
) -> Result<()> {
    Ok(())
}

/// relu(add(a, b)) as a region.
fn relu_add_region() -> PatternNode {
    PatternNode::Op {
        op: OpTag::Relu,
        attrs: OpAttrs::default(),
        operands: vec![PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
        }],
    }
}

/// relu(add(a, b)) with a downstream `neg` consumer (the reconverge).
fn graph_with_region() -> (Graph, NodeId) {
    let mut g = Graph::new();
    let s = Shape::from_dims(&[4]);
    let leaf = |g: &mut Graph| {
        g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 })
    };
    let a = leaf(&mut g);
    let b = leaf(&mut g);
    let add = g.push(Node { op: Op::Add, inputs: vec![a, b], shape: s.clone(), dtype: DType::F32 });
    let relu =
        g.push(Node { op: Op::Relu, inputs: vec![add], shape: s.clone(), dtype: DType::F32 });
    let neg = g.push(Node { op: Op::Neg, inputs: vec![relu], shape: s.clone(), dtype: DType::F32 });
    (g, neg)
}

/// Count `(Op::Branch, Op::Fused(runtime))` nodes in the graph.
fn arm_census(g: &Graph) -> (usize, usize) {
    let mut branches = 0;
    let mut runtime_fused = 0;
    for i in 0..g.len() {
        match &g.node(NodeId(i)).op {
            Op::Branch { .. } => branches += 1,
            Op::Fused(fid, FusedOpParams::Runtime { .. }) if fid.is_runtime() => {
                runtime_fused += 1
            }
            _ => {}
        }
    }
    (branches, runtime_fused)
}

fn cpu_opts() -> PlanOptions<'static> {
    PlanOptions::new().without_cost_population().with_pinned_device(DeviceLocation::Cpu)
}

#[test]
fn production_pipeline_emits_the_fused_arm_and_reset_disarms_it() {
    // LOCK DISCIPLINE (post binding-key fold): `adopt_runtime_fused` and
    // `clear_runtime_fused_for_tests` WRITE the global binding table
    // (`extend_global_bindings`), so a `global_bindings()` read guard must NOT
    // be held across them — same-thread read-then-write deadlocks. This mirrors
    // production, where adopt runs on the G7 *background* thread, never the
    // realize thread that holds read guards. Each optimize call below therefore
    // scopes its own read guard and drops it before the next adopt/reset.

    // (1) Adopt a runtime op for relu(add) on CPU (no guard held).
    let rid = adopt_runtime_fused(
        "e2e::relu_add",
        relu_add_region(),
        noop_kernel as fuel_dispatch::kernel::KernelRef,
        vec![DType::F32, DType::F32, DType::F32],
        BackendId::Cpu,
    )
    .expect("registrable region");

    // (2) The PRODUCTION entry emits the gated fused arm.
    let (mut g, root) = graph_with_region();
    {
        let bindings = fuel_dispatch::dispatch::global_bindings();
        optimize_graph_with_runtime_fusion(&mut g, &[root], &bindings, &cpu_opts())
            .expect("optimize with runtime fusion");
    }
    let (branches, fused) = arm_census(&g);
    assert_eq!(branches, 1, "one Op::Branch decision point emitted");
    assert_eq!(fused, 1, "one Op::Fused(runtime) arm emitted");
    // The fused arm is pinned to the adopted backend by the emitter.
    for i in 0..g.len() {
        if let Op::Fused(fid, FusedOpParams::Runtime { .. }) = &g.node(NodeId(i)).op {
            if *fid == rid {
                assert_eq!(
                    g.target_backend(NodeId(i)),
                    Some(BackendId::Cpu),
                    "the emitter pinned the arm's backend",
                );
            }
        }
    }

    // (3) The BARE entry never scans the sidecar — hermetic by construction.
    let (mut g2, root2) = graph_with_region();
    {
        let bindings = fuel_dispatch::dispatch::global_bindings();
        optimize_graph(&mut g2, &[root2], &bindings, &cpu_opts()).expect("bare optimize");
    }
    assert_eq!(arm_census(&g2), (0, 0), "bare optimize_graph emits no runtime arms");

    // (4) The reset hook disarms the production entry too (both the binding-table
    // RuntimeFused rows and the fuel-graph metadata sidecar cleared together; the
    // capability gate then sees no kernel). No guard held across the reset.
    clear_runtime_fused_for_tests();
    let (mut g3, root3) = graph_with_region();
    {
        let bindings = fuel_dispatch::dispatch::global_bindings();
        optimize_graph_with_runtime_fusion(&mut g3, &[root3], &bindings, &cpu_opts())
            .expect("optimize after reset");
    }
    assert_eq!(arm_census(&g3), (0, 0), "after reset there is nothing to adopt-match");

    // (5) Guard: the compile-side lookup census stayed coherent — re-adopting
    // after a reset allocates from BASE again without stale-kernel aliasing.
    let rid2 = adopt_runtime_fused(
        "e2e::relu_add::again",
        relu_add_region(),
        noop_kernel as fuel_dispatch::kernel::KernelRef,
        vec![DType::F32, DType::F32, DType::F32],
        BackendId::Cpu,
    )
    .expect("re-adopt after reset");
    assert_eq!(rid2, rid, "id allocation restarted at BASE (docs contract)");
    let mut census: HashMap<u16, usize> = HashMap::new();
    for e in fuel_graph::runtime_fused::runtime_entries() {
        *census.entry(e.id.0).or_insert(0) += 1;
    }
    assert!(census.values().all(|&c| c == 1), "no duplicate runtime ids after reset");
}

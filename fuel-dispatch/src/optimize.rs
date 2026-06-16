//! `optimize_graph` — the new "plan IS the graph" entry point.
//!
//! Phase A PR-A3a of the "plan IS the graph" rebuild
//! ([`../../docs/session-prompts/plan-is-graph-rebuild.md`](
//! ../../docs/session-prompts/plan-is-graph-rebuild.md), the PR-A3
//! line + the "Representation — DECIDED 2026-06-15" blockquote).
//!
//! ## What this is
//!
//! [`optimize_graph`] is the **new optimization entry point**: it
//! transforms a [`Graph`] *in place* into the bounded multi-path form
//! and returns a transient [`OptimizedGraph`] *view* whose dispatch
//! order is derived from `fuel-graph`'s `extract_runs` / [`lower_run`]
//! (PR-A2). The optimized form lives **in the graph** — a graph with
//! zero [`Op::Branch`] nodes is exactly today's single-route graph.
//!
//! ## A3a origin scope (ADDITIVE — proved equivalence, deleted nothing)
//!
//! This entry point landed in the **split** A3a: introduced *alongside*
//! the old path and proved equivalent before anything old was removed.
//! PR-A3b-1 then made it the default realize path, and PR-A3b-2 deleted
//! `PlanStore` and the legacy `compile_plan` route-picking dispatch
//! entirely (`optimize_graph` is now the only path). As originally
//! introduced in A3a:
//!
//! - There are **no pathfinders yet** — the first lands in PR-A4. A
//!   graph with no competing routes is already its own single-route
//!   plan, so [`optimize_graph`] introduces **zero [`Op::Branch`]
//!   nodes**. It is **single-route-only until A4**.
//! - It **reuses** the existing placement / cost / `target_backend`
//!   annotation machinery wholesale by driving
//!   [`crate::plan::compile_plan`] over the same `execution_plan`
//!   order the bridge uses — the placement DP, cost composer, filter
//!   chain, and fail-fast missing-binding diagnostics are *not*
//!   reinvented. The point of A3a is to establish the new entry point
//!   + in-place form, not new optimization.
//! - [`crate::plan::compile_plan`] + [`crate::plan::ExecutionPlan`]
//!   are reused (the latter as a transient by-product). PR-A3b-1 wired
//!   `optimize_graph` in as the default realize path; PR-A3b-2 made it
//!   the ONLY path — the legacy `compile_plan`/`PlanStore` route-picking
//!   dispatch and the identity-keyed plan store are deleted, and
//!   `optimize_graph` now surfaces its internal `ExecutionPlan` so the
//!   bridge's stamp/residency/layout passes reuse it (one `compile_plan`
//!   per realize).
//!
//! ## The equivalence gate
//!
//! For a graph with no competing routes, [`optimize_graph`] leaves
//! zero `Op::Branch` nodes and the concatenated `extract_runs` /
//! [`lower_run`] dispatch order equals today's
//! [`crate::plan::compile_plan`]`(...).order` — the exact sequence the
//! production executor walks. The order today is simply the
//! `execution_plan` topo order `compile_plan` is handed (it copies it
//! into `ExecutionPlan::order` verbatim); for a branchless,
//! single-residency graph the run extractor produces exactly one run
//! whose members are that same topo order. The gate asserts that
//! equality exactly (same `NodeId`s, same order) — see
//! [`tests::equivalence_gate_branchless_order_matches_compile_plan`].

use fuel_core_types::Result;
use fuel_graph::{extract_runs_multi, lower_run, Graph, NodeId, Op};

use crate::kernel::KernelBindingTable;
use crate::plan::{compile_plan, ExecutionPlan, PlanOptions};

/// The transient *view* [`optimize_graph`] returns — the realize-roots
/// it optimized for plus the topology generation it ran under. It is
/// **not** stored on the graph; the optimized form is the graph itself
/// (with its `Op::Branch` decision points, of which A3a emits none).
///
/// The dispatch order is derived on demand from `fuel-graph`'s run
/// extraction so the run view is always recomputed from the current
/// arena — never a stale snapshot. This is the shape PR-A3b swaps the
/// bridge onto in place of `ExecutionPlan::order`.
#[derive(Debug, Clone)]
pub struct OptimizedGraph {
    /// The realize roots this optimization targeted (the same
    /// `targets` the bridge passes today).
    pub roots: Vec<NodeId>,
    /// `SystemTopology` generation snapshotted at optimize time —
    /// mirrors [`crate::plan::ExecutionPlan::generation`] so a later
    /// chunk-boundary check can detect a topology shift exactly as
    /// the executor does today.
    pub generation: u64,
}

impl OptimizedGraph {
    /// Extract the runs of the optimized graph, in topological order
    /// of their entries (delegates to [`extract_runs_multi`]). A
    /// branchless, single-residency graph yields exactly one run.
    pub fn runs(&self, graph: &Graph) -> Vec<fuel_graph::Run> {
        extract_runs_multi(graph, &self.roots)
    }

    /// The flat executable dispatch order: the concatenation of every
    /// run's [`lower_run`] member sequence, runs in topological order.
    ///
    /// For a graph with **zero `Op::Branch` nodes** this is the exact
    /// `NodeId` sequence today's executor walks via
    /// [`crate::plan::compile_plan`]`(...).order` — the equivalence
    /// gate proves it. (Once PR-A4 introduces branches, this flat
    /// concatenation is the *single-route* lowering; the runtime
    /// picker (Phase C) selects among arms at the decision points.)
    pub fn dispatch_order(&self, graph: &Graph) -> Vec<NodeId> {
        let runs = self.runs(graph);
        let mut order = Vec::new();
        for run in &runs {
            order.extend_from_slice(lower_run(run));
        }
        order
    }

    /// The number of `Op::Branch` decision points in the arena.
    /// **Zero in A3a** (single-route-only until A4) — the contract the
    /// equivalence gate and idempotence test pin. Counting over the
    /// whole arena (not just the roots' reachable set) is the stronger
    /// claim: A3a introduces no branch anywhere.
    pub fn branch_count(&self, graph: &Graph) -> usize {
        (0..graph.len())
            .map(NodeId)
            .filter(|&id| matches!(graph.node(id).op, Op::Branch { .. }))
            .count()
    }
}

/// Optimize `graph` **in place** into the "plan IS the graph" form and
/// return the transient [`OptimizedGraph`] lowering view.
///
/// `roots` are the realize targets (the bridge's `targets`);
/// `bindings_table` is the kernel registry the placement/cost
/// machinery queries (production passes [`crate::dispatch::global_bindings`];
/// tests pass a local table); `opts` is the same [`PlanOptions`] the
/// bridge builds for `compile_plan`.
///
/// ## A3a behavior (single-route-only until A4)
///
/// With no pathfinders yet, `optimize_graph`:
///
/// 1. Derives the dispatch order via `fuel_graph::opt::execution_plan`
///    — exactly the order the bridge feeds `compile_plan` today
///    (data-flow topo refined by destructive-op ordering edges).
/// 2. Drives [`compile_plan`] over that order to **reuse** the
///    existing placement DP / cost composer / filter chain and to
///    fail-fast at build time on a missing binding or absent device
///    context (validate-at-graph-build-time per the working
///    agreement). Its [`crate::plan::ExecutionPlan`] is a transient
///    by-product here — the source of truth is the graph.
/// 3. Adds **zero** `Op::Branch` nodes — a graph with no competing
///    routes is already its own single-route plan.
///
/// The returned view's [`OptimizedGraph::dispatch_order`] (via
/// `extract_runs`/`lower_run`) reproduces `compile_plan(...).order`
/// for any branchless graph — the equivalence gate.
///
/// ## Returned `ExecutionPlan` (PR-A3b-2 de-dup)
///
/// `optimize_graph` already drives [`compile_plan`] internally for
/// placement/cost/validation; PR-A3b-2 **surfaces** that transient
/// `ExecutionPlan` alongside the [`OptimizedGraph`] so the realize
/// bridge can reuse it for its `stamp_plan_backends` / residency /
/// layout-fixup passes instead of re-running `compile_plan` a second
/// time. The plan stays a *transient by-product* — the source of truth
/// is the graph; the bridge uses the surfaced plan purely to read the
/// per-node winners it just computed. The executor still recomputes
/// the dispatch order from the (post-stamping) graph at realize time
/// (`OptimizedGraph::dispatch_order`), so the surfaced plan never
/// becomes a dispatch-time authority.
pub fn optimize_graph(
    graph: &mut Graph,
    roots: &[NodeId],
    bindings_table: &KernelBindingTable,
    opts: &PlanOptions<'_>,
) -> Result<(OptimizedGraph, ExecutionPlan)> {
    // (1) The dispatch order today: data-flow topo refined by ordering
    //     edges. `compile_plan` copies this verbatim into
    //     `ExecutionPlan::order`, so it IS the executor's walk order.
    let order = fuel_graph::opt::execution_plan(graph, roots);

    // (2) Reuse the placement/cost/validation machinery. The plan is a
    //     transient by-product — we drive `compile_plan` so the same
    //     fail-fast diagnostics (missing binding, no device context)
    //     fire at optimize time, and so the placement DP / cost composer
    //     run unchanged. We deliberately do NOT keep the plan as the
    //     source of truth (the graph is); PR-A3b-2 surfaces it back to
    //     the bridge so the bridge's stamp/residency/layout passes reuse
    //     this single `compile_plan` instead of running a second one.
    let plan = compile_plan(graph, &order, bindings_table, opts)?;
    let generation = plan.generation;

    // (3) No pathfinders in A3a => zero `Op::Branch` nodes introduced.
    //     The graph is already its own single-route plan. (This is a
    //     no-op today; it documents and pins the A3a contract — if a
    //     future edit accidentally emits a branch here, the assert and
    //     the equivalence gate both catch it.)
    debug_assert_eq!(
        (0..graph.len())
            .map(NodeId)
            .filter(|&id| matches!(graph.node(id).op, Op::Branch { .. }))
            .count(),
        graph_branch_count_before(graph, &order),
        "A3a optimize_graph must not introduce Op::Branch nodes",
    );

    Ok((
        OptimizedGraph {
            roots: roots.to_vec(),
            generation,
        },
        plan,
    ))
}

/// Count of `Op::Branch` nodes reachable in `order` — the pre-optimize
/// baseline the A3a no-new-branches assert compares against. (In A3a
/// the caller's graph is branchless, so this is 0; the helper keeps the
/// assert honest if a branch-bearing graph is ever optimized before A4
/// wires a pathfinder.)
fn graph_branch_count_before(graph: &Graph, order: &[NodeId]) -> usize {
    order
        .iter()
        .filter(|&&id| matches!(graph.node(id).op, Op::Branch { .. }))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::PrecisionGuarantee;
    use crate::kernel::{unknown_cost, KernelCaps, OpParams};
    use fuel_core_types::dispatch::OpKind;
    use fuel_core_types::probe::BackendId;
    use fuel_core_types::{DType, DeviceLocation, Layout, Result as FuelResult, Shape};
    use fuel_graph::opt::execution_plan;
    use fuel_graph::{Node, Op};
    use fuel_memory::Storage;
    use std::sync::{Arc, RwLock};

    fn noop_kernel(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> FuelResult<()> {
        Ok(())
    }

    fn register_elementwise(
        table: &mut KernelBindingTable,
        op: OpKind,
        n_in: usize,
    ) {
        let mut dtypes = vec![DType::F32; n_in];
        dtypes.push(DType::F32); // output dtype
        table.register_full(
            op,
            &dtypes,
            BackendId::Cpu,
            noop_kernel,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            unknown_cost,
        );
    }

    fn f32_node(g: &mut Graph, op: Op, inputs: Vec<NodeId>) -> NodeId {
        let id = g.push(Node {
            op,
            inputs,
            shape: Shape::from_dims(&[4]),
            dtype: DType::F32,
        });
        g.set_target_backend(id, BackendId::Cpu);
        id
    }

    /// A representative branchless CPU graph the optimizer can fully
    /// place: a fan-in (`Add` over two unary chains) feeding a unary
    /// tail. Every kernel-bearing op has a CPU binding registered. The
    /// fan-in exercises the run extractor's multi-predecessor boundary,
    /// so the equivalence holds across more than a single straight
    /// chain.
    fn build_branchless_graph(table: &mut KernelBindingTable) -> (Graph, NodeId) {
        register_elementwise(table, OpKind::ReluElementwise, 1);
        register_elementwise(table, OpKind::SiluElementwise, 1);
        register_elementwise(table, OpKind::AddElementwise, 2);
        register_elementwise(table, OpKind::TanhElementwise, 1);

        let mut g = Graph::new();
        let a = f32_node(&mut g, Op::Const, vec![]);
        let a1 = f32_node(&mut g, Op::Relu, vec![a]);
        let b = f32_node(&mut g, Op::Const, vec![]);
        let b1 = f32_node(&mut g, Op::Silu, vec![b]);
        let sum = f32_node(&mut g, Op::Add, vec![a1, b1]);
        let out = f32_node(&mut g, Op::Tanh, vec![sum]);
        (g, out)
    }

    /// A pure straight-line CPU graph — the simplest no-competing-route
    /// case (exactly one run by construction).
    fn build_straight_line_graph(table: &mut KernelBindingTable) -> (Graph, NodeId) {
        register_elementwise(table, OpKind::ReluElementwise, 1);
        register_elementwise(table, OpKind::SiluElementwise, 1);
        register_elementwise(table, OpKind::TanhElementwise, 1);

        let mut g = Graph::new();
        let a = f32_node(&mut g, Op::Const, vec![]);
        let b = f32_node(&mut g, Op::Relu, vec![a]);
        let c = f32_node(&mut g, Op::Silu, vec![b]);
        let d = f32_node(&mut g, Op::Tanh, vec![c]);
        (g, d)
    }

    fn cpu_opts() -> PlanOptions<'static> {
        // No cost machinery: A3a only needs placement/validation, not
        // ranking. Pin the realize device so every node resolves.
        PlanOptions::new()
            .without_cost_population()
            .with_pinned_device(DeviceLocation::Cpu)
    }

    /// THE EQUIVALENCE GATE (born-red until `optimize_graph` exists and
    /// its lowered order matches `compile_plan`).
    ///
    /// On representative no-competing-route graphs, assert:
    ///   (a) `optimize_graph` leaves ZERO `Op::Branch` nodes, and
    ///   (b) the concatenated `extract_runs`/`lower_run` dispatch order
    ///       EQUALS today's `compile_plan(...).order` exactly (same
    ///       NodeIds, same order).
    #[test]
    fn equivalence_gate_branchless_order_matches_compile_plan() {
        for build in [
            build_straight_line_graph as fn(&mut KernelBindingTable) -> (Graph, NodeId),
            build_branchless_graph as fn(&mut KernelBindingTable) -> (Graph, NodeId),
        ] {
            let mut table = KernelBindingTable::new();
            let (mut g, root) = build(&mut table);
            let opts = cpu_opts();

            // Today's path: the exact order the executor walks.
            let order = execution_plan(&g, &[root]);
            let plan = compile_plan(&g, &order, &table, &opts)
                .expect("today's compile_plan succeeds on a placeable graph");

            // New path.
            let (optimized, _plan) = optimize_graph(&mut g, &[root], &table, &opts)
                .expect("optimize_graph succeeds on the same graph");

            // (a) zero competing routes => zero Branch nodes.
            assert_eq!(
                optimized.branch_count(&g),
                0,
                "a no-competing-route graph optimizes to zero Op::Branch nodes",
            );

            // (b) EXACT order equality against compile_plan(...).order.
            let lowered = optimized.dispatch_order(&g);
            assert_eq!(
                lowered, plan.order,
                "the extract_runs/lower_run dispatch order must equal \
                 compile_plan(...).order exactly (same NodeIds, same order)",
            );
            // And it covers every reachable node — the executor walks
            // the whole graph, runs partition it with no gaps/dupes.
            assert_eq!(
                lowered.len(),
                order.len(),
                "the lowered order covers every node compile_plan ordered",
            );
        }
    }

    /// `optimize_graph` on a branchless graph is idempotent and adds no
    /// nodes: the node count is unchanged across repeated calls, the
    /// branch count stays zero, and the lowered order is stable.
    #[test]
    fn optimize_graph_branchless_is_idempotent_and_adds_no_nodes() {
        let mut table = KernelBindingTable::new();
        let (mut g, root) = build_branchless_graph(&mut table);
        let opts = cpu_opts();

        let nodes_before = g.len();
        let (first, _first_plan) = optimize_graph(&mut g, &[root], &table, &opts)
            .expect("first optimize succeeds");
        let order_first = first.dispatch_order(&g);
        let nodes_after_first = g.len();

        let (second, _second_plan) = optimize_graph(&mut g, &[root], &table, &opts)
            .expect("second optimize succeeds");
        let order_second = second.dispatch_order(&g);
        let nodes_after_second = g.len();

        assert_eq!(
            nodes_before, nodes_after_first,
            "optimize_graph adds no nodes (A3a is single-route-only)",
        );
        assert_eq!(
            nodes_after_first, nodes_after_second,
            "a second optimize_graph adds no further nodes — idempotent",
        );
        assert_eq!(first.branch_count(&g), 0, "no branches after first");
        assert_eq!(second.branch_count(&g), 0, "no branches after second");
        assert_eq!(
            order_first, order_second,
            "the lowered dispatch order is stable across repeated optimize",
        );
    }

    /// Build-time validation: optimize_graph fails fast (Result, never
    /// panic) when a kernel-bearing node has no registered binding —
    /// reusing compile_plan's missing-binding diagnostic.
    #[test]
    fn optimize_graph_fails_fast_on_missing_binding() {
        let table = KernelBindingTable::new(); // empty — no bindings.
        let mut g = Graph::new();
        let a = f32_node(&mut g, Op::Const, vec![]);
        let relu = f32_node(&mut g, Op::Relu, vec![a]);
        let opts = cpu_opts();

        let err = optimize_graph(&mut g, &[relu], &table, &opts)
            .map(|_| ())
            .unwrap_err();
        match err {
            fuel_core_types::Error::NoBackendForOp { op, .. } => {
                assert_eq!(op, OpKind::ReluElementwise);
            }
            other => panic!("expected NoBackendForOp, got {other:?}"),
        }
    }
}

//! `optimize_graph` — the new "plan IS the graph" entry point.
//!
//! Phase A PR-A3a/A4 of the "plan IS the graph" rebuild
//! ([`../../docs/session-prompts/plan-is-graph-rebuild.md`](
//! ../../docs/session-prompts/plan-is-graph-rebuild.md), the PR-A3/A4
//! lines + the "Representation — DECIDED 2026-06-15" blockquote, incl.
//! the architect-approved arm-0 temporary lowering).
//!
//! ## What this is
//!
//! [`optimize_graph`] is the **new optimization entry point**: it
//! transforms a [`Graph`] *in place* into the bounded multi-path form
//! and returns a transient [`OptimizedGraph`] *view* whose dispatch
//! order is the **arm-0 single-route lowering** of `fuel-graph`'s
//! `extract_runs` (PR-A2/A4). The optimized form lives **in the
//! graph** — a graph with zero [`Op::Branch`] nodes is exactly today's
//! single-route graph.
//!
//! ## PR-A4 — the first pathfinder (deliberate-fork seed)
//!
//! [`seed_placement_fork_branches`] is the first real pathfinder: where
//! the placement DP admitted a kernel-bearing node with **≥2 distinct
//! `(backend, device)` placements** that has a producer (the diverge)
//! and exactly one consumer (the reconverge), it records ONE
//! `Op::Branch` — arm-0 = the DP winner (the route realize uses today),
//! arm-1 = the runner-up clone (orphaned, read only by the Branch). It
//! emits a branch only at a genuine placement choice, never at ordinary
//! DAG fan-out, so the fewness gate holds. A CPU-only build (one
//! placement per node) emits zero branches ⇒ today's single-route graph.
//!
//! [`OptimizedGraph::dispatch_order`] is the architect-approved
//! **arm-0 single-route lowering**: it follows arm 0 through every
//! branch and skips the other arms' runs, so a branched graph realizes
//! to the same result as before (arm-0 = the winner). The Phase-C
//! runtime picker is what later selects non-arm-0 arms.
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
//! - There were **no pathfinders yet** — the first
//!   ([`seed_placement_fork_branches`]) landed in PR-A4. A graph with no
//!   competing routes is already its own single-route plan, so
//!   [`optimize_graph`] introduces **zero [`Op::Branch`] nodes** for it.
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
//! zero `Op::Branch` nodes and the `extract_runs` / `lower_run`
//! dispatch order equals today's
//! [`crate::plan::compile_plan`]`(...).order` — the exact sequence the
//! production executor walks. The order today is simply the
//! `execution_plan` topo order `compile_plan` is handed (it copies it
//! into `ExecutionPlan::order` verbatim); for a branchless,
//! single-residency graph the run extractor produces exactly one run
//! whose members are that same topo order. The gate asserts that
//! equality exactly (same `NodeId`s, same order) — see
//! [`tests::equivalence_gate_branchless_order_matches_compile_plan`].

use fuel_core_types::probe::BackendId;
use fuel_core_types::Result;
use fuel_graph::{extract_runs_multi, Graph, Node, NodeId, Op};

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

    /// The flat executable dispatch order — the **arm-0 single-route
    /// lowering** (PR-A4). It follows arm 0 through every `Op::Branch`
    /// (pre-run, arm-0's run, post-run) and **skips every non-arm-0
    /// arm's run**, via [`fuel_graph::lower_runs_arm0`].
    ///
    /// For a graph with **zero `Op::Branch` nodes** this is the exact
    /// `NodeId` sequence today's executor walks via
    /// [`crate::plan::compile_plan`]`(...).order` — the equivalence gate
    /// proves it (no arm to skip ⇒ identical to concatenating
    /// [`lower_run`] over the runs). For a branched graph it is the
    /// single-route lowering on **arm 0 = the DP winner** (the route
    /// realize used before the branch was recorded), so a branched graph
    /// realizes to the same result. The Phase-C runtime picker is what
    /// will later select non-arm-0 arms at the decision points.
    pub fn dispatch_order(&self, graph: &Graph) -> Vec<NodeId> {
        fuel_graph::lower_runs_arm0(graph, &self.roots)
    }

    /// The number of `Op::Branch` decision points in the arena. Counting
    /// over the whole arena (not just the roots' reachable set) is the
    /// stronger claim. Zero for a single-route graph (CPU-only build, or
    /// any graph with no genuine ≥2-placement fork); PR-A4's
    /// deliberate-fork pathfinder emits one per genuine placement fork.
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
/// ## Behavior
///
/// `optimize_graph`:
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
/// 3. Runs the PR-A4 deliberate-fork pathfinder
///    ([`seed_placement_fork_branches`]): records ONE `Op::Branch` per
///    genuine ≥2-placement fork (arm-0 = winner, arm-1 = runner-up). A
///    graph with no competing routes gets **zero** branches — it is
///    already its own single-route plan.
///
/// The returned view's [`OptimizedGraph::dispatch_order`] is the arm-0
/// single-route lowering; for any branchless graph it reproduces
/// `compile_plan(...).order` exactly — the equivalence gate.
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

    // (3) PR-A4: the FIRST real pathfinder. Where the placement DP found
    //     a kernel-bearing node with ≥2 viable `(device, backend)`
    //     placements diverging from a shared producer and reconverging at
    //     a forward-identical node, record ONE `Op::Branch` whose arm-0 is
    //     the DP's winner (the route realize uses today) and arm-1 the
    //     runner-up. This is the deliberate-fork SEED — branches are
    //     emitted only at genuine placement choices, never at ordinary
    //     DAG fan-out, so `branch_density` stays well under the fewness
    //     gate. A CPU-only build has one placement per node ⇒ zero forks
    //     ⇒ zero branches ⇒ today's single-route graph (unchanged).
    seed_placement_fork_branches(graph, &order, &plan)?;

    Ok((
        OptimizedGraph {
            roots: roots.to_vec(),
            generation,
        },
        plan,
    ))
}

/// PR-A4 deliberate-fork pathfinder. Scans the just-computed `plan` for
/// nodes where the placement DP / ranker admitted **≥2 distinct
/// `(backend, device)` placements** — a genuine placement choice — and
/// records each as ONE `Op::Branch` via the A1 builders.
///
/// ## What makes a node forkable (the deliberate-fork gate)
///
/// A node `fork` seeds a branch only when ALL hold:
///
/// 1. **Real ≥2-placement choice.** Its `AlternativeSet` spans two or
///    more distinct `(backend, device)` pairs. One placement (CPU-only
///    build) ⇒ no fork ⇒ no branch. This is the "deliberate fork only at
///    a real choice" rule — NOT per-node, NOT per-device-multiplicity
///    (Phase B's Pareto frontier introduces device-multiplicity arms).
/// 2. **It has a producer.** `fork` reads at least one input — that
///    input is the shared `diverge` point both arms depart from. (A
///    nullary `Op::Const` cannot diverge.)
/// 3. **It has exactly ONE consumer.** That single consumer becomes the
///    `reconverge_at` node (it already reads `fork` = arm-0's exit, so
///    the A1 arm-0-runnability invariant holds without a graph rewrite).
///    A node with ≥2 consumers is **ordinary DAG fan-out**, not a
///    decision point — a tensor feeding two different consumers is NOT a
///    branch, and the single-consumer gate also keeps arm-0's interior
///    from being read outside the branch (A1 disjointness rule 3b).
///
/// ## Arm ordering + the recording shape
///
/// - **arm-0 = the DP winner** = `fork` itself (candidate 0 after the
///   rank — the placement realize uses today). It stays exactly where it
///   is in the data flow; nothing is rewired.
/// - **arm-1 = the runner-up** = a freshly-appended clone of `fork`'s op
///   reading the same inputs, stamped onto the runner-up placement's
///   backend. It is read ONLY by the `Op::Branch` node (an orphaned
///   candidate route), so the live graph is untouched and the realize
///   result is identical (behavior-preserving).
///
/// The emitted `Op::Branch` is the in-graph *record* of the alternative;
/// the arm-0 single-route lowering ([`OptimizedGraph::dispatch_order`])
/// follows arm-0 and skips arm-1's run, so a branched graph realizes to
/// the same result as before. The Phase-C runtime picker is what will
/// later choose arm-1.
///
/// **No `DEFAULT_MAX_N` / fixed top-N anywhere here** — the bound is the
/// per-device frontier the DP already produced; this pass reads the
/// winner + the first distinct-placement runner-up off that set.
fn seed_placement_fork_branches(
    graph: &mut Graph,
    order: &[NodeId],
    plan: &ExecutionPlan,
) -> Result<()> {
    use std::collections::HashMap;

    // Consumer count over the realize-reachable set: a fork must have
    // exactly one consumer (its reconverge); ≥2 is plain fan-out.
    let mut consumer_count: HashMap<NodeId, usize> = HashMap::new();
    let mut sole_consumer: HashMap<NodeId, NodeId> = HashMap::new();
    for &id in order {
        for &input in &graph.node(id).inputs {
            *consumer_count.entry(input).or_insert(0) += 1;
            sole_consumer.insert(input, id);
        }
    }

    // Collect the fork specs first (immutable borrow of the plan), then
    // mutate the graph — `open_branch`/`finalize_branches` need `&mut`.
    struct ForkSpec {
        fork: NodeId,
        diverge: NodeId,
        reconverge: NodeId,
        runner_up_backend: BackendId,
    }
    let mut specs: Vec<ForkSpec> = Vec::new();

    for &id in order {
        let Some(set) = plan.alternatives(id) else { continue };
        // (1) A genuine ≥2-placement choice: two or more distinct
        //     (backend, device) pairs admitted by the DP/ranker.
        let winner = match set.winner() {
            Some(w) => (w.backend, w.device),
            None => continue,
        };
        let runner_up = set
            .alternatives()
            .iter()
            .map(|c| (c.backend, c.device))
            .find(|&p| p != winner);
        let Some((ru_backend, _ru_device)) = runner_up else { continue };

        // (2) A producer to serve as the shared diverge point.
        let Some(&diverge) = graph.node(id).inputs.first() else { continue };

        // (3) Exactly one consumer ⇒ deliberate fork (becomes the
        //     reconverge); ≥2 ⇒ ordinary fan-out, skip.
        if consumer_count.get(&id).copied().unwrap_or(0) != 1 {
            continue;
        }
        let reconverge = sole_consumer[&id];

        specs.push(ForkSpec {
            fork: id,
            diverge,
            reconverge,
            runner_up_backend: ru_backend,
        });
    }

    for spec in specs {
        // arm-1: a runner-up-placement clone of the fork's op reading the
        // same inputs. Orphaned (read only by the Branch) so the live
        // data flow is untouched and arm-0 = the original winner.
        let (op, inputs, shape, dtype) = {
            let n = graph.node(spec.fork);
            (n.op.clone(), n.inputs.clone(), n.shape.clone(), n.dtype)
        };
        let arm1 = graph.push(Node { op, inputs, shape, dtype });
        graph.set_target_backend(arm1, spec.runner_up_backend);

        // Record the fork as a 2-arm Branch: arm-0 = the DP winner
        // (`fork`), arm-1 = the runner-up clone. A1 validates descendant
        // reconverge, internal disjointness, uniform dtype, and arm-0
        // runnability; it never panics — a rejection surfaces as a typed
        // `Error::InvalidBranch`.
        //
        // A rejection is **non-fatal**: the branch is only a *recording*
        // of an alternative placement, so a candidate fork whose surrounding
        // graph shape happens to violate an A1 invariant (e.g. the diverge
        // reaches the reconverge by a second path, breaking disjointness) is
        // simply NOT recorded — realize proceeds on the unchanged single
        // route. The orphaned `arm1` clone left behind is unreachable from
        // any realize root, so it never dispatches and never affects the
        // result. (Build-time *correctness* diagnostics — missing binding,
        // no device — already fired inside `compile_plan` above; this gate
        // is about whether a sound branch can be *expressed*, not about
        // whether the graph is realizable.)
        let mut builder = graph.open_branch(spec.diverge);
        builder.add_arm(spec.fork); // arm-0 = winner
        builder.add_arm(arm1); // arm-1 = runner-up
        let _ = builder.finalize_branches(graph, spec.reconverge);
    }

    Ok(())
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

    // ---- PR-A4 deliberate-fork-seed test substrate ----

    fn noop_kernel_b(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> FuelResult<()> {
        Ok(())
    }

    /// Register a CPU AND a (synthetic) CUDA binding for `op` so a node
    /// enumerated against both placements gets two viable
    /// `(backend, device)` candidates — the genuine placement fork the
    /// A4 pathfinder seeds a branch from.
    fn register_two_backend(table: &mut KernelBindingTable, op: OpKind, n_in: usize) {
        let mut dtypes = vec![DType::F32; n_in];
        dtypes.push(DType::F32);
        table.register_full(
            op,
            &dtypes,
            BackendId::Cpu,
            noop_kernel,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            unknown_cost,
        );
        table.register_full(
            op,
            &dtypes,
            BackendId::Cuda,
            noop_kernel_b,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            unknown_cost,
        );
    }

    fn two_backend_placements(dev: DeviceLocation) -> Vec<BackendId> {
        if dev == DeviceLocation::Cpu {
            vec![BackendId::Cpu, BackendId::Cuda]
        } else {
            vec![]
        }
    }

    /// Two backends co-located at the realize device, so every
    /// kernel-bearing node enumerates ≥2 placements — the multi-backend
    /// build the A4 pathfinder forks on. CPU-only builds (one placement
    /// per node) never hit this; the equivalence/idempotence gates above
    /// cover that.
    fn two_backend_opts() -> PlanOptions<'static> {
        PlanOptions::new()
            .without_cost_population()
            .with_pinned_device(DeviceLocation::Cpu)
            .with_placements_for_device(&two_backend_placements)
    }

    /// A graph with a single kernel-bearing node that has a genuine
    /// ≥2-placement choice AND exactly one consumer (so it is a
    /// deliberate fork, not plain fan-out). Only `fork`'s op (`Silu`) is
    /// registered on two backends, so it is the ONLY node with a real
    /// ≥2-placement choice — exactly the "most ops are CPU-only, the
    /// matmul has cuBLAS too" shape.
    ///
    /// A long CPU-only `Relu` body surrounds the fork so the single
    /// deliberate branch sits far below the fewness threshold (a tiny
    /// graph would be all-branch by node count): `c -> body* -> prod ->
    /// fork -> tail -> body*`. `prod` is the diverge; `tail` is the
    /// reconverge (it reads `fork`).
    fn build_single_fork_graph(
        table: &mut KernelBindingTable,
    ) -> (Graph, NodeId, NodeId, NodeId, NodeId) {
        register_elementwise(table, OpKind::ReluElementwise, 1);
        register_two_backend(table, OpKind::SiluElementwise, 1); // the fork
        register_elementwise(table, OpKind::TanhElementwise, 1);

        let mut g = Graph::new();
        let mut prev = f32_node(&mut g, Op::Const, vec![]);
        // Straight-line CPU body before the fork.
        for _ in 0..20 {
            prev = f32_node(&mut g, Op::Relu, vec![prev]);
        }
        let prod = prev; // the diverge (fork's producer)
        let fork = f32_node(&mut g, Op::Silu, vec![prod]);
        let tail = f32_node(&mut g, Op::Tanh, vec![fork]); // reconverge
        // Straight-line CPU body after the reconverge — the realize root
        // is the body's tail, so the whole graph is reachable and the
        // single fork sits far below the fewness threshold.
        let mut prev = tail;
        for _ in 0..20 {
            prev = f32_node(&mut g, Op::Relu, vec![prev]);
        }
        let root = prev;
        (g, prod, fork, tail, root)
    }

    /// THE A4 DELIBERATE-FORK GATE (born-red until the pathfinder
    /// emits a Branch).
    ///
    /// (a) A node with two viable `(backend, device)` placements and a
    /// single consumer ⇒ `optimize_graph` emits exactly ONE 2-arm
    /// `Op::Branch` whose arm-0 is the DP winner, and the result passes
    /// the fewness gate.
    #[test]
    fn deliberate_fork_emits_one_two_arm_branch() {
        use fuel_graph::{branch_density, passes_fewness_gate};
        let mut table = KernelBindingTable::new();
        let (mut g, _prod, fork, _tail, root) = build_single_fork_graph(&mut table);
        let opts = two_backend_opts();

        let nodes_before = g.len();
        let (optimized, _plan) = optimize_graph(&mut g, &[root], &table, &opts)
            .expect("optimize_graph succeeds on a 2-placement graph");

        // Exactly one Op::Branch in the arena.
        assert_eq!(
            optimized.branch_count(&g),
            1,
            "a single genuine placement fork emits exactly one Op::Branch",
        );

        // It is a 2-arm branch (arm-0 = the winner = `fork`).
        let branch_id = (nodes_before..g.len())
            .map(NodeId)
            .find(|&id| matches!(g.node(id).op, Op::Branch { .. }))
            .expect("a Branch node was appended");
        let branch = g.node(branch_id);
        assert_eq!(branch.inputs.len(), 2, "the branch has exactly two arms");
        assert_eq!(
            branch.inputs[0], fork,
            "arm-0 is the DP winner node (the route realize uses today)",
        );
        assert_ne!(
            branch.inputs[1], fork,
            "arm-1 is a distinct (runner-up placement) node",
        );

        // The fewness gate holds — one branch among many nodes.
        assert!(
            passes_fewness_gate(&g, root),
            "a single deliberate fork passes the fewness gate; density={}",
            branch_density(&g, root),
        );
    }

    /// (b) An ordinary DAG fan-out (one result, two distinct consumers)
    /// with the SAME 2-placement freedom is NOT flagged as a branch —
    /// fan-out is not a decision point.
    #[test]
    fn plain_fan_out_is_not_a_branch() {
        let mut table = KernelBindingTable::new();
        // Only the fan-out node's op (`Relu`) is dual-backend, so it is
        // the sole ≥2-placement candidate — and it is excluded purely
        // because it fans out (two consumers), not because it lacks a
        // placement choice. The consumers are CPU-only.
        register_two_backend(&mut table, OpKind::ReluElementwise, 1);
        register_elementwise(&mut table, OpKind::SiluElementwise, 1);
        register_elementwise(&mut table, OpKind::TanhElementwise, 1);
        register_elementwise(&mut table, OpKind::AddElementwise, 2);

        // `shared` has two distinct consumers (c0, c1) that join at
        // `out` — plain fan-out, not a fork — even though it has a real
        // 2-placement choice.
        let mut g = Graph::new();
        let c = f32_node(&mut g, Op::Const, vec![]);
        let shared = f32_node(&mut g, Op::Relu, vec![c]);
        let c0 = f32_node(&mut g, Op::Silu, vec![shared]);
        let c1 = f32_node(&mut g, Op::Tanh, vec![shared]);
        let out = f32_node(&mut g, Op::Add, vec![c0, c1]);
        let opts = two_backend_opts();

        let (optimized, _plan) = optimize_graph(&mut g, &[out], &table, &opts)
            .expect("optimize_graph succeeds");
        assert_eq!(
            optimized.branch_count(&g),
            0,
            "plain fan-out (multiple consumers) is not a deliberate fork",
        );
        let _ = shared;
    }

    /// (c) A branched graph realizes correctly via arm-0: the dispatch
    /// order follows {pre, arm-0, post} and SKIPS arm-1's run.
    #[test]
    fn branched_graph_realizes_via_arm0() {
        let mut table = KernelBindingTable::new();
        let (mut g, prod, fork, tail, root) = build_single_fork_graph(&mut table);
        let opts = two_backend_opts();

        // Capture the single-route order BEFORE the branch is emitted —
        // this is the order realize must reproduce on arm-0.
        let pre_order = execution_plan(&g, &[root]);

        let (optimized, _plan) = optimize_graph(&mut g, &[root], &table, &opts)
            .expect("optimize_graph succeeds");
        assert_eq!(optimized.branch_count(&g), 1, "exactly one branch");

        // The arm-1 node is the branch's second input — it must NOT
        // appear in the arm-0 dispatch order.
        let branch_id = (0..g.len())
            .map(NodeId)
            .find(|&id| matches!(g.node(id).op, Op::Branch { .. }))
            .expect("a Branch node exists");
        let arm1 = g.node(branch_id).inputs[1];

        let order = optimized.dispatch_order(&g);
        assert!(
            !order.contains(&arm1),
            "arm-0 lowering must skip arm-1's run; order={order:?} arm1={arm1:?}",
        );
        // arm-0 (fork) and the pre/post nodes ARE executed.
        assert!(order.contains(&prod), "pre-run (diverge producer) executes");
        assert!(order.contains(&fork), "arm-0 (the winner) executes");
        assert!(order.contains(&tail), "post-run (reconverge) executes");
        assert!(order.contains(&root), "the realize root executes");

        // Behavior preserving: arm-0 lowering reproduces exactly the
        // pre-branch single-route order (arm-0 = the DP winner = the
        // graph realize used before the branch was recorded).
        assert_eq!(
            order, pre_order,
            "arm-0 dispatch order equals the pre-branch single-route order",
        );
    }

    /// (d) No `DEFAULT_MAX_N` truncation: the fork records the winner +
    /// a runner-up without a fixed top-N cap stranding placements. Both
    /// arms survive; the per-device frontier — not a fixed N — bounds
    /// the arms.
    #[test]
    fn no_default_max_n_truncation() {
        let mut table = KernelBindingTable::new();
        let (mut g, _prod, fork, _tail, root) = build_single_fork_graph(&mut table);
        let opts = two_backend_opts();

        let (optimized, _plan) = optimize_graph(&mut g, &[root], &table, &opts)
            .expect("optimize_graph succeeds");
        assert_eq!(optimized.branch_count(&g), 1);
        let branch_id = (0..g.len())
            .map(NodeId)
            .find(|&id| matches!(g.node(id).op, Op::Branch { .. }))
            .expect("a Branch node exists");
        // Two distinct placements ⇒ two arms survive; the runner-up was
        // not dropped by any top-N cap.
        assert_eq!(g.node(branch_id).inputs.len(), 2);
        assert_eq!(g.node(branch_id).inputs[0], fork);
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

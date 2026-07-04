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

use std::collections::{HashMap, HashSet};

use fuel_ir::probe::BackendId;
use fuel_ir::{DeviceLocation, Result};
use fuel_graph::opt::insert_cross_device_copies;
use fuel_graph::{extract_runs_multi, topo_order_multi, Graph, NodeId, Op};

use crate::driver::{OptimizationContext, PassRegistry};
use crate::kernel::KernelBindingTable;
use crate::plan::{compile_plan, ExecutionPlan, PlanOptions};
use crate::topology::SystemTopology;

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
/// ## The internal `ExecutionPlan` (Step D — no longer returned)
///
/// `optimize_graph` drives [`compile_plan`] internally for
/// placement/cost/validation; the resulting `ExecutionPlan` is a
/// **transient, optimizer-internal** accumulator the stamp / residency /
/// layout-fixup passes + the lock-step rankers read below. It is NOT
/// returned or threaded anywhere — the durable output is the graph's
/// `target_backend` stamps + `Op::Branch` arms. The executor reads the
/// graph + re-derives any per-arm candidate from the binding registry at
/// realize time, so the plan never becomes a dispatch-time authority.
pub fn optimize_graph(
    graph: &mut Graph,
    roots: &[NodeId],
    bindings_table: &KernelBindingTable,
    opts: &PlanOptions<'_>,
) -> Result<OptimizedGraph> {
    // (1) The dispatch order today: data-flow topo refined by ordering
    //     edges. `compile_plan` copies this verbatim into
    //     `ExecutionPlan::order`, so it IS the executor's walk order.
    let order = fuel_graph::opt::execution_plan(graph, roots);

    // (2) Reuse the placement/cost/validation machinery. The plan is a
    //     transient, optimizer-INTERNAL by-product — we drive `compile_plan`
    //     so the same fail-fast diagnostics (missing binding, no device
    //     context) fire at optimize time, and so the placement DP / cost
    //     composer run unchanged. The plan is the working accumulator the
    //     stamp / residency / layout-fixup passes + the lock-step rankers
    //     read below; it is NOT returned or threaded (Step D — the graph
    //     stamps + arms are the only durable output).
    let plan = compile_plan(graph, &order, bindings_table, opts)?;
    let generation = plan.generation;

    // (3) PR-B3: the lock-step pass driver. The pre-B3 hardcoded
    //     `seed_placement_fork_branches(...)` call is replaced by a
    //     registry of pathfinders + optimizers run lock-step
    //     (prune-as-you-go): for each registered pathfinder, ADD its
    //     candidate paths, then immediately run every registered
    //     optimizer to MERGE/DISCARD over the region just touched. The
    //     shipped configuration ([`PassRegistry::default_passes`]) is the
    //     PR-A4 `PlacementForkPathfinder` (the deliberate-fork seed) +
    //     the `FrontierConvergenceOptimizer` (duplicate-path convergence
    //     + the never-strand / no-active-cycle invariant guards) — which
    //     is exactly the pre-B3 sequence, re-expressed. A CPU-only build
    //     has one placement per node ⇒ the pathfinder proposes zero
    //     branches ⇒ today's single-route graph (unchanged).
    //
    //     The *ranker* + the PR-B2 per-device Pareto frontier are applied
    //     per kernel-bearing node *inside* `compile_plan` (the MEASURE +
    //     per-node PRUNE), and the ranked, frontier-pruned result is the
    //     `plan` the driver reads via `OptimizationContext::plan`. Batch
    //     optimize has no executing region, so the cycle guard is empty.
    let cycle_guard: HashSet<NodeId> = HashSet::new();
    let ctx = OptimizationContext {
        order: &order,
        plan: &plan,
        cycle_guard: &cycle_guard,
    };
    PassRegistry::default_passes().run_lockstep(graph, &ctx)?;

    // "Plan IS the graph": commit the placement decision INTO the graph's
    // `target_backend` side-table here, so a fully-optimized graph carries
    // its own backend stamps and downstream consumers read the graph, not a
    // threaded plan. Guarded on a pinned device — the production realize
    // path always sets one (`PlanOptions::with_pinned_device`); bare-graph
    // test callers that don't are unaffected. (Cleanup Step A: this is
    // idempotent with the bridge's transitional `stamp_plan_backends`, which
    // re-stamps the identical result; Step A2 removes the bridge copy.)
    if let Some(pinned) = opts.pinned_device {
        stamp_plan_backends(graph, roots, &plan, pinned);
        // Optimize-time kernel-variant bake: resolve **same-device**
        // kernel-variant branches (a decomposed region vs. a fused kernel on
        // the SAME device — e.g. the CUDA flash decode arm) to their cost
        // winner and COLLAPSE them, so the winning variant is selectable
        // without a runtime pick (04-optimization: kernel-variant choice is
        // "largely baked at optimize time"). Runs AFTER `stamp_plan_backends`
        // (the arms now carry `target_backend`, which the same-device gate
        // reads) and BEFORE `insert_residency_copies` (so the residency pass
        // stitches inbound copies only for the surviving winner arm, not a
        // pruned one). A **placement** branch (arms on ≥2 devices) is left
        // LIVE for the runtime route picker; a graph with no same-device
        // variant branch — every CPU/Vulkan build today, since no CUDA flash
        // arm is offered there — is a strict no-op. Ties / unknown costs /
        // capability-missing default to arm 0 (the decomposed oracle).
        crate::variant_bake::bake_variant_branches(graph, roots, &|g, branch, arm, interior| {
            let backend = g.target_backend(g.node(branch).inputs.first().copied()?)?;
            crate::variant_bake::decode_arm_composite_ns(g, branch, arm, interior, backend)
        });
        // Cleanup Step B (residency): the optimizer-side cross-device copy pass
        // (was the bridge's `insert_resident_input_copies`). Runs AFTER stamping
        // (it reads `target_backend`) and BEFORE layout-fixup, preserving the
        // prior residency→layout order. Cache residency (where persistent /
        // const inputs already live) arrives via the `opts.input_residency`
        // provider the realize path already threads in — the optimizer reads
        // that fact, it does not touch realize-time storage.
        insert_residency_copies(graph, roots, &plan, pinned, opts.input_residency);
        // Cleanup Step B (layout): insert `Op::Contiguize` before any kernel
        // whose chosen winner rejects strided inputs and whose input layout is
        // non-contiguous — the optimizer writing the layout-fixup decision INTO
        // the graph (was the bridge's `apply_layout_fixups`, now retired). No
        // runtime deps: reads only the in-scope `plan` winner caps + graph
        // layout. CSE-deduped + idempotent. Gated on `pinned_device` so it runs
        // on the realize path (which always sets one), matching the prior
        // bridge behavior; bare-graph test callers without a pin are unaffected.
        fuel_graph::opt::insert_layout_fixups(graph, roots, |id| {
            plan.alternatives(id)
                .and_then(|set| set.winner())
                .map(|cand| cand.caps.strided_input)
                .unwrap_or(true)
        });
    }

    Ok(OptimizedGraph {
        roots: roots.to_vec(),
        generation,
    })
}

/// Commit the plan's per-node winner backend to the graph's
/// `target_backend` side-table — the optimizer writing its placement
/// decision into the graph.
///
/// Per kernel-bearing node: stamp `winner.backend` if the plan has an
/// `AlternativeSet` for it, else the pinned device's backend (structural
/// ops the planner skips — `Op::Copy`/`Op::Move`/`Op::Alloc`/`Op::ZeroFill`
/// — plus any op without an `OpKind` mapping). `Op::Const`/`Op::Release`/
/// `Op::Contiguize`/view ops/`Op::Reshape` inherit or don't need a stamp
/// and are skipped. Always overwrites, so re-optimizing after a
/// `TopologyChanged` retry re-stamps consistently from the fresh plan.
fn stamp_plan_backends(
    graph: &mut Graph,
    roots: &[NodeId],
    plan: &ExecutionPlan,
    pinned_loc: DeviceLocation,
) {
    let pinned_backend = location_to_backend_id(pinned_loc);
    let order = topo_order_multi(graph, roots);
    for &id in &order {
        let node = graph.node(id);
        if matches!(node.op, Op::Const | Op::Release | Op::Contiguize)
            || node.op.is_view_op()
            || matches!(node.op, Op::Reshape(_))
        {
            continue;
        }
        let stamp = plan
            .alternatives(id)
            .and_then(|set| set.winner())
            .map(|c| c.backend)
            .unwrap_or(pinned_backend);
        graph.set_target_backend(id, stamp);
    }
}

/// Cleanup Step B (residency): the optimizer-side cross-device copy pass —
/// insert `Op::Copy { target }` on every edge whose producer's resident
/// location doesn't share a storage substrate with the consumer's placement,
/// then stamp each inserted copy's `target_backend` with the SOURCE backend
/// (the pipelined executor's `Op::Copy` convention: the transfer kernel runs
/// on the backend the bytes come FROM). Was the bridge's
/// `insert_resident_input_copies`; moved here so the optimizer — not the
/// realize-time bridge — owns the residency decision and writes it into the
/// graph.
///
/// Placements come from graph-knowable facts only (no realize-time storage):
/// residency-declaring ops, explicit `Graph::placement` (the bridge converts
/// cache residency into placement stamps before optimize — cleanup Step B),
/// the plan winner's device, the `target_backend` stamp (→ pinned), and view
/// pass-through. Substrate sharing is queried via the process-global
/// `SystemTopology` (available pre-realize once backends are registered).
///
/// Runs after `stamp_plan_backends` (it reads `target_backend`); the re-stamp
/// sweep below restores the source-backend stamp on copies/moves that the
/// stamping pass overwrote with the pinned backend.
fn insert_residency_copies(
    graph: &mut Graph,
    roots: &[NodeId],
    plan: &ExecutionPlan,
    pinned_loc: DeviceLocation,
    input_residency: Option<&dyn Fn(NodeId) -> Option<DeviceLocation>>,
) {
    // Step E Phase C, PR C-0: walk the ARM-INCLUSIVE reachable set, not bare
    // `roots`. A finalized `Op::Branch` is orphaned (its `reconverge_at` reads
    // arm-0 directly, per the PR-A1 runnability invariant), so a plain
    // `topo_order_multi(roots)` never reaches it — and therefore never reaches
    // its non-arm-0 arms. The residency pass MUST stitch inbound cross-device
    // copies for EVERY surviving arm so the executor can legally re-pick
    // arm-1+ at runtime by live load (C2): whichever arm it picks, that arm's
    // device-inputs are already resident. `effective_roots` (the same seeding
    // the run extractor + route picker use) pulls every Branch's arms into the
    // walk. On a branchless graph it returns exactly `roots` ⇒ byte-identical.
    let eff_roots = fuel_graph::effective_roots(graph, roots);
    let placements = effective_placements(graph, &eff_roots, plan, pinned_loc, input_residency);
    let topology = SystemTopology::current();
    // Identical locations short-circuit to `true` BEFORE the topology lookup
    // (a location trivially shares bytes with itself; `shares_storage` returns
    // `false` for unknown backends, so without this an unprobed topology would
    // wrap every same-device edge in a copy).
    let shares = |a: DeviceLocation, b: DeviceLocation| -> bool {
        if a == b {
            return true;
        }
        topology.shares_storage(
            (location_to_backend_id(a), a),
            (location_to_backend_id(b), b),
        )
    };
    // Hop-count oracle: an edge whose transfer path is `HostStaging` can't
    // be done as a single `Op::Copy` (no direct device-to-device copy
    // kernel between distinct GPU substrates — the cross-VENDOR CUDA↔Vulkan
    // case), so the residency pass routes it through a CPU intermediate as
    // TWO hops. `SystemTopology::transfer_path` already returns `HostStaging`
    // for such pairs and `SameDevice` for `a == b`; CPU↔CUDA / CPU↔Vulkan
    // resolve to a direct path (or the universal host-staging fallback only
    // when CPU is NOT one of the endpoints), so any CPU-touching edge stays
    // single-hop. We additionally guard on neither endpoint being CPU: a CPU
    // intermediate for a CPU-touching edge would be nonsensical (and the
    // single-hop CPU↔GPU Copy kernels exist and are byte-identical to
    // today). Same-vendor GPU pairs that don't share storage (e.g. CUDA gpu0
    // ↔ gpu1) fall back to `HostStaging` too — correctly two-hop, since
    // there's no direct cross-gpu_id copy kernel today either.
    let needs_host_staging = |a: DeviceLocation, b: DeviceLocation| -> bool {
        if a == DeviceLocation::Cpu || b == DeviceLocation::Cpu {
            return false;
        }
        matches!(topology.transfer_path(a, b), fuel_ir::backend::TransferPath::HostStaging)
    };
    let inserted = insert_cross_device_copies(
        graph,
        &eff_roots,
        |id| placements.get(&id).copied(),
        shares,
        needs_host_staging,
    );
    // Source-location resolver for stamping inserted copies. The
    // `placements` snapshot was computed BEFORE insertion, so it does NOT
    // contain the freshly-inserted CPU intermediate of a two-hop edge. That
    // intermediate's placement was written into the graph by the pass
    // (`set_placement(cpu_id, Cpu)`), so fall back to `graph.placement(src)`
    // when the snapshot lacks the node — this is what stamps a two-hop's
    // consumer-side hop with `Cpu` (its bytes come FROM the CPU intermediate,
    // so its Copy kernel is the CPU `copy_from_cpu_wrapper`).
    let src_location = |graph: &Graph, src: NodeId| -> Option<DeviceLocation> {
        placements.get(&src).copied().or_else(|| graph.placement(src))
    };
    // Stamp the new copies: target_backend = SOURCE backend. The pass only
    // inserts a copy when the producer's placement resolved to Some.
    for &copy_id in &inserted {
        if let Some(&src) = graph.node(copy_id).inputs.first() {
            if let Some(src_loc) = src_location(graph, src) {
                graph.set_target_backend(copy_id, location_to_backend_id(src_loc));
            }
        }
    }
    // Re-stamp ALL copies/moves with their SOURCE backend — graph rewrites are
    // sticky and `stamp_plan_backends` just overwrote pre-existing ones (e.g.
    // realize-root splices) with the pinned backend; the transfer kernel runs
    // where the bytes come from. The freshly-inserted copies are visited too,
    // but their input's placement resolves to the same source location, so the
    // sweep re-applies the identical stamp (idempotent). Copies whose source
    // placement is absent keep whatever stamp they already have. The two-hop
    // CPU intermediate's source is the original GPU producer (stamped to that
    // GPU); the consumer-side hop's source is the CPU intermediate (stamped
    // Cpu via the `graph.placement` fallback in `src_location`).
    // Arm-inclusive (C-0): re-walk from the arm-seeded roots so copies just
    // inserted inside non-arm-0 arms are re-stamped too. The `eff_roots` seed
    // set (Branch nodes are unchanged by insertion) reaches the fresh arm
    // copies via the now-rewired arm-input edges.
    let order = topo_order_multi(graph, &eff_roots);
    for &id in &order {
        if !matches!(graph.node(id).op, Op::Copy { .. } | Op::Move { .. }) {
            continue;
        }
        let Some(&src) = graph.node(id).inputs.first() else { continue };
        let Some(src_loc) = src_location(graph, src) else { continue };
        graph.set_target_backend(id, location_to_backend_id(src_loc));
    }
}

/// Compute every reachable node's *effective placement* for the residency
/// pass, mirroring the bridge's old `effective_placements`. Priority per node:
///
/// 1. **Residency-declaring ops** — `Op::Copy`/`Op::Move`/`Op::Alloc` carry
///    their output location in the variant (definitional).
/// 2. **Explicit `Graph::placement`** — set by inserted copies (and any
///    caller-provided placement hints).
/// 3. **Input residency** — where a persistent / const input already lives,
///    supplied by the realize path via `PlanOptions::input_residency`. This is
///    the only runtime fact, read through the provider closure — the optimizer
///    never touches the storage itself.
/// 4. **Plan winner** — a planned node runs on its winner's device.
/// 5. **Backend stamp** — a node with `target_backend` but no plan entry
///    (structural ops) follows the pinned device.
/// 6. **View pass-throughs** — view ops / `Reshape` / `Contiguize` follow
///    their data input (already resolved; `order` is topological).
fn effective_placements(
    g: &Graph,
    roots: &[NodeId],
    plan: &ExecutionPlan,
    pinned_loc: DeviceLocation,
    input_residency: Option<&dyn Fn(NodeId) -> Option<DeviceLocation>>,
) -> HashMap<NodeId, DeviceLocation> {
    let order = topo_order_multi(g, roots);
    let mut map: HashMap<NodeId, DeviceLocation> =
        HashMap::with_capacity(order.len());
    for &id in &order {
        let node = g.node(id);
        match node.op {
            Op::Copy { target } | Op::Move { target } | Op::Alloc { target } => {
                map.insert(id, target);
                continue;
            }
            _ => {}
        }
        if let Some(loc) = g.placement(id) {
            map.insert(id, loc);
            continue;
        }
        if let Some(loc) = input_residency.and_then(|f| f(id)) {
            map.insert(id, loc);
            continue;
        }
        if let Some(winner) = plan.alternatives(id).and_then(|set| set.winner()) {
            map.insert(id, winner.device);
            continue;
        }
        if g.target_backend(id).is_some() {
            map.insert(id, pinned_loc);
            continue;
        }
        if node.op.is_view_op()
            || matches!(node.op, Op::Reshape(_) | Op::Contiguize)
        {
            if let Some(&loc) = node.inputs.first().and_then(|i| map.get(i)) {
                map.insert(id, loc);
            }
        }
    }
    map
}

fn location_to_backend_id(loc: DeviceLocation) -> BackendId {
    match loc {
        DeviceLocation::Cpu => BackendId::Cpu,
        DeviceLocation::Cuda { .. } => BackendId::Cuda,
        DeviceLocation::Vulkan { .. } => BackendId::Vulkan,
        DeviceLocation::Metal { .. } => BackendId::Metal,
    }
}

/// The PR-A4 deliberate-fork pathfinder + the PR-B2 frontier
/// convergence optimizer now live in [`crate::driver`] as registered
/// [`crate::driver::Pathfinder`] / [`crate::driver::Optimizer`] impls
/// ([`crate::driver::PlacementForkPathfinder`] /
/// [`crate::driver::FrontierConvergenceOptimizer`]). PR-B3 replaced the
/// hardcoded `seed_placement_fork_branches(...)` call in
/// [`optimize_graph`] with a lock-step
/// [`crate::driver::PassRegistry::run_lockstep`] drive over those
/// registered passes (see the module docs). The pathfinder body is
/// unchanged — it moved verbatim behind the trait.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::PrecisionGuarantee;
    use crate::kernel::{unknown_cost, KernelCaps, OpParams};
    use fuel_ir::dispatch::OpKind;
    use fuel_ir::probe::BackendId;
    use fuel_ir::{DType, DeviceLocation, Layout, Result as FuelResult, Shape};
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
        // No pre-stamp: post-Step-A `optimize_graph` writes `target_backend`
        // itself, so pre-stamping here is redundant for the tests that run the
        // optimizer — and it would falsify the `*_stamps_backends_*` tests'
        // "unstamped before optimize" / "Const leaf not stamped" preconditions.
        g.push(Node {
            op,
            inputs,
            shape: Shape::from_dims(&[4]),
            dtype: DType::F32,
        })
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

    /// Cleanup Step A: `optimize_graph` stamps each computational node's
    /// chosen backend onto the graph (the optimizer writes its placement
    /// decision INTO the graph). On a pinned-CPU straight-line graph every
    /// kernel-bearing node is stamped `Cpu`; the leaf `Op::Const` is
    /// skipped (it inherits / needs no stamp). When `pinned_device` is
    /// unset, no stamping occurs (the bare-graph test contract).
    #[test]
    fn optimize_graph_stamps_backends_onto_the_graph() {
        let mut table = KernelBindingTable::new();
        let (mut g, root) = build_straight_line_graph(&mut table);
        // a=Const(0), b=Relu(1), c=Silu(2), d=Tanh(3) by construction.
        let (a, b, c, d) = (NodeId(0), NodeId(1), NodeId(2), NodeId(3));
        assert!(g.target_backend(b).is_none(), "unstamped before optimize");

        let _optimized = optimize_graph(&mut g, &[root], &table, &cpu_opts())
            .expect("optimize_graph on a straight-line CPU graph");

        assert_eq!(g.target_backend(b), Some(BackendId::Cpu), "Relu stamped Cpu");
        assert_eq!(g.target_backend(c), Some(BackendId::Cpu), "Silu stamped Cpu");
        assert_eq!(g.target_backend(d), Some(BackendId::Cpu), "Tanh stamped Cpu");
        assert!(g.target_backend(a).is_none(), "Op::Const leaf is not stamped");
    }

    /// The stamping is guarded on a pinned device — an `optimize_graph`
    /// with no `pinned_device` leaves the graph unstamped (so the bare
    /// optimize.rs test callers below are unaffected by Step A).
    #[test]
    fn optimize_graph_without_pinned_device_does_not_stamp() {
        let mut table = KernelBindingTable::new();
        let (mut g, root) = build_straight_line_graph(&mut table);
        // Give each node a device context via a graph PLACEMENT (a distinct
        // side-table from `target_backend`), so `compile_plan` resolves without
        // a pinned device — then assert `optimize_graph` performs no
        // `target_backend` stamping (the stamping arm is guarded on
        // `pinned_device`, so without a pin nothing is stamped).
        for id in [NodeId(0), NodeId(1), NodeId(2), NodeId(3)] {
            g.set_placement(id, DeviceLocation::Cpu);
        }
        let opts = PlanOptions::new().without_cost_population(); // no pinned device
        let _ = optimize_graph(&mut g, &[root], &table, &opts).expect("optimize_graph");
        assert!(
            g.target_backend(NodeId(1)).is_none(),
            "no pinned device ⇒ no target_backend stamping",
        );
    }

    /// Cleanup Step B (layout): `optimize_graph` inserts an `Op::Contiguize`
    /// before a kernel whose chosen winner rejects strided inputs
    /// (`caps.strided_input == false`, e.g. the `KernelCaps::empty()`
    /// elementwise bindings) and whose input layout is non-contiguous — the
    /// layout-fixup pass moved from the realize-time bridge into the optimizer.
    /// The pass itself is exhaustively tested in `fuel-graph::opt`; this guards
    /// the optimizer-side wiring (callback reads the plan winner's caps, runs
    /// inside the pinned guard).
    #[test]
    fn optimize_graph_inserts_layout_fixup_for_strided_input() {
        let mut table = KernelBindingTable::new();
        register_elementwise(&mut table, OpKind::ReluElementwise, 1); // empty caps ⇒ rejects strided
        let mut g = Graph::new();
        let a = f32_node(&mut g, Op::Const, vec![]);
        let r = f32_node(&mut g, Op::Relu, vec![a]);
        // Make `a`'s effective layout non-contiguous (non-zero start_offset),
        // so the strided-rejecting Relu needs a Contiguize on its input.
        g.set_layout(a, Layout::contiguous_with_offset(Shape::from_dims(&[4]), 1));

        let _optimized = optimize_graph(&mut g, &[r], &table, &cpu_opts())
            .expect("optimize_graph with a strided input");

        // Relu's input was `a`; after the fixup it must read through an inserted
        // Op::Contiguize wrapping `a`.
        let relu_input = g.node(r).inputs[0];
        assert!(
            matches!(g.node(relu_input).op, Op::Contiguize),
            "strided input to a strided-rejecting kernel must be fixed up via \
             Op::Contiguize, got {:?}",
            g.node(relu_input).op,
        );
        assert_eq!(
            g.node(relu_input).inputs,
            vec![a],
            "the inserted Op::Contiguize wraps the original strided input",
        );
    }

    // ---- Cleanup Step B (residency) — insert_residency_copies tests ----
    // (migrated from the bridge's insert_resident_input_copies unit tests;
    // cache residency is supplied via an `input_residency` closure instead of a
    // live StorageCache. Placement metadata only — no GPU needed; the unprobed
    // test `SystemTopology` reports Cpu/Cuda as non-sharing, so a crossing is
    // detected and a copy inserted.)

    /// Co-located graph ⇒ `insert_residency_copies` is a no-op (no crossings).
    #[test]
    fn residency_noop_when_colocated() {
        let mut g = Graph::new();
        let c1 = f32_node(&mut g, Op::Const, vec![]);
        let c2 = f32_node(&mut g, Op::Const, vec![]);
        let add = f32_node(&mut g, Op::Add, vec![c1, c2]);
        g.set_target_backend(add, BackendId::Cpu);
        let residency =
            |id: NodeId| (id == c1 || id == c2).then_some(DeviceLocation::Cpu);
        let pre = g.len();
        insert_residency_copies(
            &mut g, &[add], &ExecutionPlan::empty(), DeviceLocation::Cpu,
            Some(&residency),
        );
        assert_eq!(g.len(), pre, "co-located graph must be a no-op");
        assert_eq!(g.node(add).inputs, vec![c1, c2], "edges untouched");
    }

    /// A CPU-resident input feeding two CUDA-pinned consumers ⇒ exactly ONE
    /// `Op::Copy` bridges the crossing (CSE-deduped), targeting the consumer
    /// device, stamped `target_backend = SOURCE` (Cpu, the H2D wrapper) with
    /// its output placed on the consumer device.
    #[test]
    fn residency_one_copy_per_crossing_deduped() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let mut g = Graph::new();
        let c1 = f32_node(&mut g, Op::Const, vec![]);
        let neg = f32_node(&mut g, Op::Neg, vec![c1]);
        let sqr = f32_node(&mut g, Op::Sqr, vec![c1]);
        g.set_target_backend(neg, BackendId::Cuda);
        g.set_target_backend(sqr, BackendId::Cuda);
        let residency = |id: NodeId| (id == c1).then_some(DeviceLocation::Cpu);

        let pre = g.len();
        insert_residency_copies(
            &mut g, &[neg, sqr], &ExecutionPlan::empty(), cuda0, Some(&residency),
        );

        assert_eq!(g.len(), pre + 1, "one crossing → one copy, CSE-deduped");
        let neg_in = g.node(neg).inputs[0];
        let sqr_in = g.node(sqr).inputs[0];
        assert_eq!(neg_in, sqr_in, "both consumers share the one copy");
        assert_ne!(neg_in, c1, "consumers rewired off the raw input");
        let copy = g.node(neg_in);
        assert!(
            matches!(copy.op, Op::Copy { target } if target == cuda0),
            "copy targets the consumer device; got {:?}", copy.op,
        );
        assert_eq!(copy.inputs, vec![c1], "copy reads the resident slot");
        assert_eq!(
            g.target_backend(neg_in), Some(BackendId::Cpu),
            "stamped with the SOURCE backend (H2D runs on the CPU wrapper)",
        );
        assert_eq!(
            g.placement(neg_in), Some(cuda0),
            "copy output placed on the consumer device",
        );
    }

    /// Idempotence + re-stamp: a second pass on the rewritten graph inserts
    /// nothing and restores the source-backend stamp even after a clobber
    /// (mimicking `stamp_plan_backends` overwriting it with the pinned backend).
    #[test]
    fn residency_idempotent_and_restamped_on_second_call() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let mut g = Graph::new();
        let c1 = f32_node(&mut g, Op::Const, vec![]);
        let neg = f32_node(&mut g, Op::Neg, vec![c1]);
        g.set_target_backend(neg, BackendId::Cuda);
        let residency = |id: NodeId| (id == c1).then_some(DeviceLocation::Cpu);

        insert_residency_copies(
            &mut g, &[neg], &ExecutionPlan::empty(), cuda0, Some(&residency),
        );
        let copy_id = g.node(neg).inputs[0];
        // Simulate stamp_plan_backends clobbering the copy with the pinned backend.
        g.set_target_backend(copy_id, BackendId::Cuda);

        let pre = g.len();
        insert_residency_copies(
            &mut g, &[neg], &ExecutionPlan::empty(), cuda0, Some(&residency),
        );
        assert_eq!(g.len(), pre, "re-run inserts nothing");
        assert_eq!(
            g.target_backend(copy_id), Some(BackendId::Cpu),
            "re-stamp sweep restores the source-backend stamp",
        );
    }

    // ---- Step E Phase C, PR C-0: arm residency-copy audit ----
    //
    // C2 lets the executor re-pick an `Op::Branch` arm at runtime by live
    // device load — it may choose arm-1+ instead of the static arm-0 winner.
    // That is only legal if the optimizer's residency pass has ALREADY
    // stitched the inbound cross-device `Op::Copy` for EVERY surviving arm,
    // so whichever arm the executor picks, that arm's device-inputs are
    // resident on its device. This test builds a 2-arm diamond whose two arms
    // live on DIFFERENT devices (arm-0 CPU, arm-1 CUDA) sharing one CPU
    // producer, runs the residency pass, and asserts BOTH arms' cross-device
    // inputs got their copy — not just arm-0's.
    //
    // The finalized `Op::Branch` is orphaned (its `reconverge_at` reads arm-0
    // directly, per the PR-A1 runnability invariant), so arm-1 is reachable
    // ONLY through the Branch node. `insert_residency_copies` walks
    // `topo_order_multi(roots)` directly (NOT `effective_roots`), so it never
    // reaches an orphaned Branch — and therefore never reaches arm-1. The
    // run/route machinery (`extract_runs_multi`, `branches_in_topo_order`)
    // does use `effective_roots`; the residency pass does not. This is the
    // C-0 prerequisite gap.

    /// Build a 2-arm diamond: a CPU producer `diverge` fans into arm-0 (CPU)
    /// and arm-1 (`arm1_loc`); `reconverge` reads arm-0 (the runnability
    /// invariant); the Branch merges {arm0, arm1} at `reconverge`. Returns
    /// `(graph, branch, diverge, arm0, arm1, post)`. Placements are set via
    /// `set_placement` so the residency pass reads them with no plan/winner.
    fn residency_diamond(
        arm1_loc: DeviceLocation,
    ) -> (Graph, NodeId, NodeId, NodeId, NodeId, NodeId) {
        let mut g = Graph::new();
        let pre = f32_node(&mut g, Op::Const, vec![]);
        let diverge = f32_node(&mut g, Op::Relu, vec![pre]);
        let arm0 = f32_node(&mut g, Op::Silu, vec![diverge]);
        let arm1 = f32_node(&mut g, Op::Gelu, vec![diverge]);
        let reconverge = f32_node(&mut g, Op::Relu, vec![arm0]);
        let mut b = g.open_branch(diverge);
        b.add_arm(arm0);
        b.add_arm(arm1);
        let branch = b
            .finalize_branches(&mut g, reconverge)
            .expect("well-formed 2-arm branch")
            .expect("2 arms survive");
        let post = f32_node(&mut g, Op::Tanh, vec![reconverge]);
        // Producer + arm-0 + the post-merge region on CPU; arm-1 elsewhere.
        g.set_placement(pre, DeviceLocation::Cpu);
        g.set_placement(diverge, DeviceLocation::Cpu);
        g.set_placement(arm0, DeviceLocation::Cpu);
        g.set_placement(reconverge, DeviceLocation::Cpu);
        g.set_placement(post, DeviceLocation::Cpu);
        g.set_placement(arm1, arm1_loc);
        (g, branch, diverge, arm0, arm1, post)
    }

    /// C-0 PREREQUISITE: the residency pass must insert the inbound
    /// cross-device `Op::Copy` for BOTH arms of a 2-device branch — arm-0
    /// (CPU, same device as the producer ⇒ no copy needed) AND arm-1 (CUDA,
    /// crosses ⇒ MUST get a copy). If only arm-0 were stitched, re-picking
    /// arm-1 at runtime (C2) would dispatch a CUDA kernel whose input was
    /// never copied to the GPU → the un-bridged-mixed-edge error / a UAF.
    #[test]
    fn residency_stitches_every_branch_arm_not_just_arm0() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let (mut g, _branch, diverge, arm0, arm1, post) = residency_diamond(cuda0);

        insert_residency_copies(
            &mut g, &[post], &ExecutionPlan::empty(), DeviceLocation::Cpu, None,
        );

        // arm-0 is CPU, same substrate as the CPU producer ⇒ its input edge
        // does NOT cross ⇒ no copy (and must stay reading `diverge`).
        assert_eq!(
            g.node(arm0).inputs, vec![diverge],
            "arm-0 is co-located with its producer ⇒ no copy, edge untouched",
        );

        // arm-1 is CUDA, its producer `diverge` is CPU ⇒ the edge crosses ⇒
        // the residency pass MUST have inserted an Op::Copy targeting CUDA on
        // arm-1's input. THIS is the C-0 invariant: every surviving arm's
        // device-inputs are made resident, not just arm-0's.
        let arm1_in = g.node(arm1).inputs[0];
        assert_ne!(
            arm1_in, diverge,
            "arm-1 (CUDA) reads the CPU producer directly — its inbound \
             cross-device Op::Copy was NEVER inserted. Re-picking arm-1 at \
             runtime (C2) would dispatch a CUDA kernel over un-copied CPU \
             bytes. The residency pass stitched only arm-0's route.",
        );
        let copy = g.node(arm1_in);
        assert!(
            matches!(copy.op, Op::Copy { target } if target == cuda0),
            "arm-1's inbound copy must target CUDA; got {:?}", copy.op,
        );
        assert_eq!(
            copy.inputs, vec![diverge],
            "arm-1's copy reads the shared CPU producer",
        );
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
        let optimized = optimize_graph(&mut g, &[root], &table, &opts)
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

        let optimized = optimize_graph(&mut g, &[out], &table, &opts)
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

        let optimized = optimize_graph(&mut g, &[root], &table, &opts)
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

        let optimized = optimize_graph(&mut g, &[root], &table, &opts)
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
            let optimized = optimize_graph(&mut g, &[root], &table, &opts)
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
        let first = optimize_graph(&mut g, &[root], &table, &opts)
            .expect("first optimize succeeds");
        let order_first = first.dispatch_order(&g);
        let nodes_after_first = g.len();

        let second = optimize_graph(&mut g, &[root], &table, &opts)
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
            fuel_ir::Error::NoBackendForOp { op, .. } => {
                assert_eq!(op, OpKind::ReluElementwise);
            }
            other => panic!("expected NoBackendForOp, got {other:?}"),
        }
    }

    // ===== Phase B PR-B3: the lock-step pathfinder/ranker/optimizer
    //        driver. =====

    use crate::driver::{
        FrontierConvergenceOptimizer, OptimizationContext, Optimizer, PassRegistry,
        Pathfinder, PlacementForkPathfinder,
    };
    use std::collections::HashSet;

    /// Snapshot of a graph's `Op::Branch` decision points, keyed by
    /// arena order: `(branch_id, reconverge_at, arm NodeIds)`. Two
    /// optimization paths that produce the same snapshot produced the
    /// same multi-path graph (behavior-preservation).
    fn branch_snapshot(g: &Graph) -> Vec<(NodeId, NodeId, Vec<NodeId>)> {
        (0..g.len())
            .map(NodeId)
            .filter_map(|id| match g.node(id).op {
                Op::Branch { reconverge_at } => {
                    Some((id, reconverge_at, g.node(id).inputs.clone()))
                }
                _ => None,
            })
            .collect()
    }

    /// THE LOAD-BEARING BEHAVIOR-PRESERVATION GATE (born-red until
    /// `optimize_graph` routes through the lock-step driver).
    ///
    /// On a representative multi-placement graph, the driver
    /// (`PassRegistry::default_passes().run_lockstep`) produces the
    /// **identical** multi-path graph as the old hardcoded sequence
    /// (pathfinder then optimizer, in that order): same branch count,
    /// same arm-0 winner, same arms, same reconverge points, and the
    /// same arm-0 dispatch order. B3 is a structural refactor — the
    /// optimized result must not change.
    #[test]
    fn driver_matches_legacy_sequence() {
        // (A) The reference: drive the registered passes BY HAND in the
        //     legacy order (propose, then prune) onto a freshly compiled
        //     plan — exactly what the pre-B3 hardcoded sequence did.
        let mut ref_table = KernelBindingTable::new();
        let (mut ref_g, _prod, ref_fork, _tail, ref_root) =
            build_single_fork_graph(&mut ref_table);
        let opts = two_backend_opts();
        let ref_order = execution_plan(&ref_g, &[ref_root]);
        let ref_plan = compile_plan(&ref_g, &ref_order, &ref_table, &opts)
            .expect("reference compile_plan succeeds");
        {
            let guard: HashSet<NodeId> = HashSet::new();
            let ctx = OptimizationContext {
                order: &ref_order,
                plan: &ref_plan,
                cycle_guard: &guard,
            };
            // Legacy order: the pathfinder ADDs, then the optimizer
            // prunes — the same two operations the hardcoded sequence ran.
            PlacementForkPathfinder
                .propose(&mut ref_g, &ctx)
                .expect("pathfinder proposes");
            FrontierConvergenceOptimizer
                .prune(&mut ref_g, &ctx)
                .expect("optimizer prunes");
        }
        let ref_snapshot = branch_snapshot(&ref_g);
        let ref_dispatch = fuel_graph::lower_runs_arm0(&ref_g, &[ref_root]);

        // (B) The driver path: optimize_graph routes through
        //     PassRegistry::default_passes().run_lockstep.
        let mut table = KernelBindingTable::new();
        let (mut g, _p, fork, _t, root) = build_single_fork_graph(&mut table);
        let optimized = optimize_graph(&mut g, &[root], &table, &opts)
            .expect("optimize_graph succeeds");
        let got_snapshot = branch_snapshot(&g);
        let got_dispatch = optimized.dispatch_order(&g);

        // Identical multi-path graph: same branches, arms, reconverge.
        assert_eq!(
            got_snapshot, ref_snapshot,
            "driver produces the identical Op::Branch structure as the legacy \
             pathfinder→optimizer sequence",
        );
        assert_eq!(
            got_snapshot.len(),
            1,
            "the representative multi-placement graph produces exactly one branch",
        );
        assert_eq!(
            got_snapshot[0].2[0], fork,
            "arm-0 is the DP winner (the route realize uses)",
        );
        // Identical dispatch order (arm-0 single-route lowering).
        assert_eq!(
            got_dispatch, ref_dispatch,
            "driver yields the identical arm-0 dispatch order as the legacy sequence",
        );
        let _ = ref_fork;
    }

    /// A no-op registry (no registered pathfinders) leaves a branchless
    /// graph unchanged: no nodes added, zero branches, identical
    /// dispatch order. Proves the driver itself introduces no structure.
    #[test]
    fn noop_registry_leaves_branchless_graph_unchanged() {
        let mut table = KernelBindingTable::new();
        let (mut g, root) = build_branchless_graph(&mut table);
        let opts = cpu_opts();

        let order = execution_plan(&g, &[root]);
        let plan = compile_plan(&g, &order, &table, &opts)
            .expect("compile_plan succeeds");

        let dispatch_before = fuel_graph::lower_runs_arm0(&g, &[root]);
        let nodes_before = g.len();

        let registry = PassRegistry::new(); // empty — no passes.
        assert_eq!(registry.pathfinder_count(), 0);
        assert_eq!(registry.optimizer_count(), 0);

        let guard: HashSet<NodeId> = HashSet::new();
        let ctx = OptimizationContext {
            order: &order,
            plan: &plan,
            cycle_guard: &guard,
        };
        registry
            .run_lockstep(&mut g, &ctx)
            .expect("empty registry runs cleanly");

        assert_eq!(g.len(), nodes_before, "no-op registry adds no nodes");
        let branches = (0..g.len())
            .map(NodeId)
            .filter(|&id| matches!(g.node(id).op, Op::Branch { .. }))
            .count();
        assert_eq!(branches, 0, "no-op registry adds no branches");
        let dispatch_after = fuel_graph::lower_runs_arm0(&g, &[root]);
        assert_eq!(
            dispatch_before, dispatch_after,
            "no-op registry leaves the dispatch order unchanged",
        );
    }

    /// Lock-step ordering: a pathfinder ADDs before its dependent
    /// optimizer PRUNEs. We register a recording pathfinder and a
    /// recording optimizer that each stamp a shared log when invoked;
    /// `run_lockstep` must invoke the pathfinder strictly before the
    /// optimizer (prune-as-you-go, never optimizer-first).
    #[test]
    fn lockstep_runs_pathfinder_before_dependent_optimizer() {
        use std::sync::{Arc, Mutex};

        #[derive(Clone)]
        struct RecordPathfinder(Arc<Mutex<Vec<&'static str>>>);
        impl Pathfinder for RecordPathfinder {
            fn name(&self) -> &'static str {
                "RecordPathfinder"
            }
            fn propose(
                &self,
                _g: &mut Graph,
                _ctx: &OptimizationContext<'_>,
            ) -> fuel_ir::Result<()> {
                self.0.lock().unwrap().push("propose");
                Ok(())
            }
        }
        #[derive(Clone)]
        struct RecordOptimizer(Arc<Mutex<Vec<&'static str>>>);
        impl Optimizer for RecordOptimizer {
            fn name(&self) -> &'static str {
                "RecordOptimizer"
            }
            fn prune(
                &self,
                _g: &mut Graph,
                _ctx: &OptimizationContext<'_>,
            ) -> fuel_ir::Result<()> {
                self.0.lock().unwrap().push("prune");
                Ok(())
            }
        }

        let log = Arc::new(Mutex::new(Vec::new()));
        let registry = PassRegistry::new()
            .with_pathfinder(Box::new(RecordPathfinder(log.clone())))
            .with_optimizer(Box::new(RecordOptimizer(log.clone())));

        let mut g = Graph::new();
        let _a = f32_node(&mut g, Op::Const, vec![]);
        let plan = ExecutionPlan::empty();
        let order: Vec<NodeId> = Vec::new();
        let guard: HashSet<NodeId> = HashSet::new();
        let ctx = OptimizationContext {
            order: &order,
            plan: &plan,
            cycle_guard: &guard,
        };
        registry.run_lockstep(&mut g, &ctx).expect("drive runs");

        let seq = log.lock().unwrap().clone();
        assert_eq!(
            seq,
            vec!["propose", "prune"],
            "lock-step drive runs the pathfinder ADD strictly before its \
             dependent optimizer PRUNE (prune-as-you-go)",
        );
    }
}

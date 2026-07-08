//! The `optimize_graph` lock-step pass driver — Phase B PR-B3 of the
//! "plan IS the graph" rebuild
//! ([`../../docs/session-prompts/plan-is-graph-rebuild.md`](
//! ../../docs/session-prompts/plan-is-graph-rebuild.md) capability [5];
//! [`../../docs/architecture/04-optimization.md`](
//! ../../docs/architecture/04-optimization.md) §"The two-stage
//! transformation", §"Relationship to PR 3").
//!
//! ## What this is
//!
//! `optimize_graph` was a hardcoded sequence — `execution_plan` order →
//! [`compile_plan`](crate::plan::compile_plan) (per-node placement +
//! cost ranking + the PR-B2 per-ending-device Pareto frontier) →
//! `seed_placement_fork_branches` (the PR-A4 deliberate-fork
//! pathfinder). PR-B3 restructures that sequence into a **lock-step
//! driver** over three registered pass *kinds*, mirroring the shape of
//! `fuel_graph::opt::RuleRegistry` (a `Vec<Box<dyn …>>` of passes +
//! builder + a driver method that walks them) so future passes (fusion,
//! algebraic, dtype-lowering) plug in without rewriting the loop:
//!
//! - **[`Pathfinder`]s** — *ADD* candidate paths / branches to the
//!   graph. The PR-A4 [`PlacementForkPathfinder`] (deliberate-fork seed,
//!   the one pathfinder that records an `Op::Branch` per genuine
//!   multi-placement fork) is the first registered pathfinder.
//! - **Rankers** — *MEASURE* each path on the `CostVector` (PR-B1/B2
//!   `rank_by_cost`). Rankers are not a driver-visible trait in B3:
//!   ranking is applied **per kernel-bearing node inside
//!   [`compile_plan`]** as the per-node `AlternativeSet` is built, and
//!   the ranked result is carried into the driver via
//!   [`OptimizationContext::plan`]. The driver re-runs no ranker; the
//!   measure step is the plan it is handed. (B3 is a structural refactor
//!   — when a later PR adds a pathfinder that emits *new* candidates the
//!   plan never priced, that is the point to lift ranking into a
//!   driver-visible `Ranker` trait.)
//! - **[`Optimizer`]s** — *MERGE / DISCARD* paths. The PR-B2
//!   per-ending-device Pareto frontier + crowding cap is applied
//!   per-node inside [`compile_plan`]; the registered
//!   [`FrontierConvergenceOptimizer`] is the in-graph counterpart that
//!   runs **after each pathfinder** to (a) collapse duplicate
//!   (forward-identical) arms a pathfinder may have proposed
//!   [duplicate-path convergence], and (b) assert the optimizer
//!   invariants the constitution requires — never strand the last
//!   `(device, backend)` path, never mutate a node in an active cycle /
//!   currently-executing region.
//!
//! ## Lock-step / prune-as-you-go (the constitution's bound)
//!
//! The driver runs the passes interleaved, **not** explode-then-extract:
//! for each registered pathfinder, it ADDs its candidate paths, then
//! immediately runs every registered optimizer to prune the region the
//! pathfinder just touched, before moving to the next pathfinder. This
//! is the [`PassRegistry::run_lockstep`] contract — the working set
//! never explodes-then-extracts, so it stays bounded by construction
//! (per [`../../docs/architecture/04-optimization.md`](
//! ../../docs/architecture/04-optimization.md) §"Bounding the
//! frontier": *"run the merge/discard optimizers after each pathfinder
//! step, not once at the end"*).
//!
//! ## Behavior preservation (B3 is a structural refactor)
//!
//! The optimized result — the arm-0 winner, the retained per-device
//! frontier (applied inside `compile_plan`, unchanged), and the emitted
//! `Op::Branch` nodes — is **identical** to the pre-B3 hardcoded
//! sequence on every graph. The driver is the same three operations in
//! the same order; the only change is that they are reached through the
//! registry rather than inlined. The load-bearing
//! `driver_matches_legacy_sequence` test in
//! [`crate::optimize`] proves it on a representative multi-placement
//! graph.

use std::collections::{HashMap, HashSet};

use fuel_ir::probe::BackendId;
use fuel_ir::Result;
use fuel_graph::{Graph, Node, NodeId, Op};

use crate::plan::ExecutionPlan;

/// Read-only context threaded to every [`Pathfinder::propose`] and
/// [`Optimizer::prune`] call during a [`PassRegistry::run_lockstep`]
/// drive.
///
/// It carries the **measure** stage's output — the ranked, per-device
/// frontier-pruned per-node `AlternativeSet`s [`compile_plan`] produced
/// ([`Self::plan`]) plus the dispatch [`Self::order`] — and the
/// **incremental-reopt guard** ([`Self::cycle_guard`]): the set of nodes
/// that lie on an active cycle or in a currently-executing region and so
/// must never be mutated by an optimizer. Batch `optimize_graph` has no
/// executing region, so the guard is empty there; it exists to document
/// and enforce the incremental-reopt contract for the Phase-C runtime.
///
/// [`compile_plan`]: crate::plan::compile_plan
pub struct OptimizationContext<'a> {
    /// The dispatch order [`compile_plan`](crate::plan::compile_plan)
    /// was driven over — data-flow topo refined by destructive-op
    /// ordering edges.
    pub order: &'a [NodeId],
    /// The per-node ranked + frontier-pruned `AlternativeSet`s (the
    /// *measure* + per-node *prune* output). Pathfinders read the
    /// winner + runner-up placements off this; optimizers read it to
    /// honor the never-strand-last invariant.
    pub plan: &'a ExecutionPlan,
    /// Nodes on an active cycle / currently-executing region — an
    /// optimizer must not mutate any node in this set. Empty for batch
    /// `optimize_graph`.
    pub cycle_guard: &'a HashSet<NodeId>,
}

/// A **pathfinder** *ADDs* candidate paths / branches to the graph. The
/// PR-A4 deliberate-fork seed ([`PlacementForkPathfinder`]) is the first
/// registered pathfinder; later pathfinders (fusion, algebraic,
/// dtype-lowering under tolerance, layout fixups) implement the same
/// trait and register the same way.
///
/// `propose` may append nodes and record `Op::Branch` decision points
/// (via the `fuel-graph` A1 builders). It must never *remove* a path —
/// that is an [`Optimizer`]'s job — and it returns `Result` so a
/// build-time failure surfaces (never panic, per the working agreement).
pub trait Pathfinder: Send + Sync {
    /// Stable, human-readable name. Shows up in debug traces.
    fn name(&self) -> &'static str;

    /// Add candidate paths to `graph`. `ctx` carries the ranked plan +
    /// dispatch order the *measure* stage produced.
    fn propose(&self, graph: &mut Graph, ctx: &OptimizationContext<'_>) -> Result<()>;
}

/// An **optimizer** *MERGEs / DISCARDs* paths. The PR-B2 per-device
/// Pareto frontier is applied per-node inside
/// [`compile_plan`](crate::plan::compile_plan); the registered
/// [`FrontierConvergenceOptimizer`] is its in-graph counterpart that
/// runs after each pathfinder to collapse duplicate (forward-identical)
/// arms and enforce the optimizer invariants.
///
/// `prune` runs **after** each [`Pathfinder::propose`] (lock-step), so
/// the working set never explodes-then-extracts. It must honor the
/// optimizer invariants: never strand the last `(device, backend)` path,
/// and never mutate a node in `ctx.cycle_guard`.
pub trait Optimizer: Send + Sync {
    /// Stable, human-readable name. Shows up in debug traces.
    fn name(&self) -> &'static str;

    /// Merge/discard candidate paths in `graph`. Runs after each
    /// pathfinder over the region that pathfinder just touched.
    fn prune(&self, graph: &mut Graph, ctx: &OptimizationContext<'_>) -> Result<()>;
}

/// A registry of [`Pathfinder`]s and [`Optimizer`]s, driven lock-step by
/// [`Self::run_lockstep`]. Mirrors `fuel_graph::opt::RuleRegistry`: a
/// `Vec<Box<dyn …>>` of passes built up with `with_*` builders and run
/// through one driver method. Keeping the same shape means the future
/// passes the constitution names (fusion, algebraic, dtype-lowering)
/// register exactly as the two shipped passes do.
#[derive(Default)]
pub struct PassRegistry {
    pathfinders: Vec<Box<dyn Pathfinder>>,
    optimizers: Vec<Box<dyn Optimizer>>,
}

impl PassRegistry {
    /// Empty registry. Use [`Self::with_pathfinder`] /
    /// [`Self::with_optimizer`] to add passes, or [`Self::default_passes`]
    /// for the shipped configuration.
    pub fn new() -> Self {
        Self {
            pathfinders: Vec::new(),
            optimizers: Vec::new(),
        }
    }

    /// Append a pathfinder. Returns self for builder-style chaining.
    pub fn with_pathfinder(mut self, p: Box<dyn Pathfinder>) -> Self {
        self.pathfinders.push(p);
        self
    }

    /// Append an optimizer. Returns self for builder-style chaining.
    pub fn with_optimizer(mut self, o: Box<dyn Optimizer>) -> Self {
        self.optimizers.push(o);
        self
    }

    /// The shipped Phase-B configuration: the PR-A4
    /// [`PlacementForkPathfinder`] (the one pathfinder that records an
    /// `Op::Branch` per genuine multi-placement fork) and the
    /// [`FrontierConvergenceOptimizer`] (duplicate-path convergence +
    /// the per-device-frontier invariant guards). This is exactly the
    /// pre-B3 hardcoded sequence re-expressed as registered passes.
    pub fn default_passes() -> Self {
        Self::new()
            .with_pathfinder(Box::new(PlacementForkPathfinder))
            .with_optimizer(Box::new(FrontierConvergenceOptimizer))
    }

    /// [`Self::default_passes`] **plus runtime fusion**: the
    /// [`crate::runtime_fused_pathfinder::RuntimeFusedArmPathfinder`] runs
    /// FIRST (before placement — placement must see the fused arms it emits;
    /// the fork seed can't double-fork them because `ctx.plan`/`ctx.order`
    /// predate the drive and don't contain the appended arm nodes).
    ///
    /// A separate constructor rather than a change to `default_passes` so the
    /// bare registry never scans the process-global runtime-fused sidecar —
    /// non-adopting tests stay hermetic *by construction*, not by reset
    /// discipline (dd-shapes coordination, 2026-07-08). Transitional, like the
    /// sidecar itself: both collapse when runtime entries fold into the one
    /// binding registry (10-decisions-log, runtime-fused-sidecar entry).
    pub fn default_passes_with_runtime_fusion() -> Self {
        Self::new()
            .with_pathfinder(Box::new(
                crate::runtime_fused_pathfinder::RuntimeFusedArmPathfinder,
            ))
            .with_pathfinder(Box::new(PlacementForkPathfinder))
            .with_optimizer(Box::new(FrontierConvergenceOptimizer))
    }

    /// Number of registered pathfinders.
    pub fn pathfinder_count(&self) -> usize {
        self.pathfinders.len()
    }

    /// Number of registered optimizers.
    pub fn optimizer_count(&self) -> usize {
        self.optimizers.len()
    }

    /// Drive the passes **lock-step / prune-as-you-go**: for each
    /// registered pathfinder, ADD its candidate paths, then immediately
    /// run every registered optimizer to MERGE/DISCARD over the region
    /// that pathfinder just touched, before moving to the next
    /// pathfinder.
    ///
    /// This is the constitution's bound: the merge/discard optimizers
    /// run **after each pathfinder step, not once at the end**, so the
    /// working set never explodes-then-extracts. Ordering is enforced by
    /// construction — an optimizer can only ever see the paths the
    /// pathfinders *before it in this drive* added.
    ///
    /// Returns `Result`; the first failing pass aborts the drive
    /// (build-time validation, never a panic).
    pub fn run_lockstep(
        &self,
        graph: &mut Graph,
        ctx: &OptimizationContext<'_>,
    ) -> Result<()> {
        for pf in &self.pathfinders {
            // (1) ADD.
            pf.propose(graph, ctx)?;
            // (2) PRUNE — every optimizer, over the region just touched.
            //     Lock-step: never deferred to a final batch pass.
            for opt in &self.optimizers {
                opt.prune(graph, ctx)?;
            }
        }
        Ok(())
    }
}

/// The PR-A4 deliberate-fork pathfinder, re-expressed as a registered
/// [`Pathfinder`]. Scans the ranked plan for nodes the placement DP /
/// ranker admitted with **≥2 distinct `(backend, device)` placements** —
/// a genuine placement choice — and records each as ONE `Op::Branch` via
/// the `fuel-graph` A1 builders (arm-0 = the DP winner, arm-1 = the
/// runner-up clone, orphaned so the live data flow is untouched and the
/// realize result is identical).
///
/// This is **the same logic** the pre-B3 `seed_placement_fork_branches`
/// free function ran — moved verbatim behind the trait. The
/// deliberate-fork gate (real ≥2-placement choice, a producer to
/// diverge from, exactly one consumer to reconverge at) is unchanged, so
/// the fewness gate still holds and a CPU-only build emits zero branches.
///
/// **No `DEFAULT_MAX_N` / fixed top-N anywhere** — the bound is the
/// per-device frontier the plan already carries; this pass reads the
/// winner + the first distinct-placement runner-up off that set.
pub struct PlacementForkPathfinder;

impl Pathfinder for PlacementForkPathfinder {
    fn name(&self) -> &'static str {
        "PlacementForkSeed"
    }

    fn propose(&self, graph: &mut Graph, ctx: &OptimizationContext<'_>) -> Result<()> {
        // Consumer count over the realize-reachable set: a fork must
        // have exactly one consumer (its reconverge); ≥2 is plain
        // fan-out, not a decision point.
        let mut consumer_count: HashMap<NodeId, usize> = HashMap::new();
        let mut sole_consumer: HashMap<NodeId, NodeId> = HashMap::new();
        for &id in ctx.order {
            for &input in &graph.node(id).inputs {
                *consumer_count.entry(input).or_insert(0) += 1;
                sole_consumer.insert(input, id);
            }
        }

        // Collect the fork specs first (immutable borrow of the plan),
        // then mutate the graph — `open_branch`/`finalize_branches` need
        // `&mut`.
        struct ForkSpec {
            fork: NodeId,
            diverge: NodeId,
            reconverge: NodeId,
            runner_up_backend: BackendId,
        }
        let mut specs: Vec<ForkSpec> = Vec::new();

        for &id in ctx.order {
            let Some(set) = ctx.plan.alternatives(id) else {
                continue;
            };
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
            let Some((ru_backend, _ru_device)) = runner_up else {
                continue;
            };

            // (2) A producer to serve as the shared diverge point.
            let Some(&diverge) = graph.node(id).inputs.first() else {
                continue;
            };

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
            // arm-1: a runner-up-placement clone of the fork's op reading
            // the same inputs. Orphaned (read only by the Branch) so the
            // live data flow is untouched and arm-0 = the original winner.
            let (op, inputs, shape, dtype) = {
                let n = graph.node(spec.fork);
                (n.op.clone(), n.inputs.clone(), n.shape.clone(), n.dtype)
            };
            let arm1 = graph.push(Node {
                op,
                inputs,
                shape,
                dtype,
            });
            graph.set_target_backend(arm1, spec.runner_up_backend);

            // Record the fork as a 2-arm Branch: arm-0 = the DP winner
            // (`fork`), arm-1 = the runner-up clone. A1 validates
            // descendant reconverge, internal disjointness, uniform
            // dtype, and arm-0 runnability; it never panics — a rejection
            // surfaces as a typed `Error::InvalidBranch`.
            //
            // A rejection is **non-fatal**: the branch is only a
            // *recording* of an alternative placement, so a candidate
            // fork whose surrounding graph shape happens to violate an A1
            // invariant is simply NOT recorded — realize proceeds on the
            // unchanged single route. The orphaned `arm1` clone left
            // behind is unreachable from any realize root, so it never
            // dispatches and never affects the result.
            let mut builder = graph.open_branch(spec.diverge);
            builder.add_arm(spec.fork); // arm-0 = winner
            builder.add_arm(arm1); // arm-1 = runner-up
            let _ = builder.finalize_branches(graph, spec.reconverge);
        }

        Ok(())
    }
}

/// The PR-B2 per-device-frontier optimizer, re-expressed as a registered
/// [`Optimizer`]. It runs **after** each pathfinder (lock-step) and is
/// the in-graph MERGE/DISCARD counterpart of
/// [`AlternativeSet::retain_per_device_frontier`](
/// crate::ranker::AlternativeSet::retain_per_device_frontier), which is
/// applied per kernel-bearing node *inside*
/// [`compile_plan`](crate::plan::compile_plan).
///
/// Its two responsibilities:
///
/// 1. **Duplicate-path convergence (the trivial MERGE optimizer).**
///    Detect any two arms of a just-emitted `Op::Branch` that are
///    *forward-identical* — same op, same inputs, same target backend.
///    Two pathfinders (or one pathfinder firing twice) could in
///    principle propose structurally-identical routes; the constitution
///    requires *"convergence merges only forward-identical paths"*. The
///    PR-A4 seed proposes one runner-up per fork, distinct from arm-0 by
///    construction (different `target_backend`), so today no duplicate
///    arm ever arises and this detection is a no-op — it is the trivial
///    second optimizer the spec calls for, in place (with a debug
///    assertion that it stays a no-op) so the contract is documented and
///    enforced before a future multi-runner-up pathfinder can leak a
///    duplicate arm into a branch.
///
/// 2. **Invariant guard (the never-strand / no-active-cycle contract).**
///    Assert that every `Op::Branch` keeps ≥1 viable `(device, backend)`
///    route — never strand the last path (upheld by the per-node frontier
///    inside `compile_plan`, re-asserted here over the in-graph form) —
///    and that no branch the optimizer would touch is in `ctx.cycle_guard`
///    (the incremental-reopt contract — batch optimize has an empty
///    guard, but the guard documents and enforces the rule for the
///    Phase-C runtime).
pub struct FrontierConvergenceOptimizer;

impl FrontierConvergenceOptimizer {
    /// Forward-identical key of an arm: `(op, inputs, target_backend)`.
    /// Two arms with equal keys are the same route and would be merged
    /// by duplicate-path convergence.
    fn arm_key(graph: &Graph, id: NodeId) -> (String, Vec<NodeId>, Option<BackendId>) {
        let n = graph.node(id);
        (
            format!("{:?}", n.op),
            n.inputs.clone(),
            graph.target_backend(id),
        )
    }

    /// Count of *distinct* (forward-identical-deduplicated) arms among
    /// `arms`. arm-0 always counts; a later arm equal to an earlier one
    /// does not. Equal to `arms.len()` exactly when every arm is a
    /// distinct route.
    fn distinct_arm_count(graph: &Graph, arms: &[NodeId]) -> usize {
        let mut seen: Vec<(String, Vec<NodeId>, Option<BackendId>)> = Vec::new();
        for &arm in arms {
            let k = Self::arm_key(graph, arm);
            if !seen.iter().any(|s| *s == k) {
                seen.push(k);
            }
        }
        seen.len()
    }
}

impl Optimizer for FrontierConvergenceOptimizer {
    fn name(&self) -> &'static str {
        "FrontierConvergence"
    }

    fn prune(&self, graph: &mut Graph, ctx: &OptimizationContext<'_>) -> Result<()> {
        // Scan the in-graph branches. The per-device Pareto frontier
        // proper is applied per kernel-bearing node *inside* compile_plan
        // (PR-B2); here we run its in-graph counterpart — duplicate-path
        // convergence + the never-strand / no-active-cycle invariant
        // guards.
        let n = graph.len();
        let mut branches: Vec<(NodeId, Vec<NodeId>)> = Vec::new();
        for i in 0..n {
            let id = NodeId(i);
            if matches!(graph.node(id).op, Op::Branch { .. }) {
                branches.push((id, graph.node(id).inputs.clone()));
            }
        }

        for (branch_id, arms) in branches {
            // Invariant guard (incremental-reopt contract): never touch a
            // branch whose decision point or any arm is on an active
            // cycle / currently-executing region. Batch optimize has an
            // empty guard, so this never trips here; it enforces the
            // contract for the Phase-C runtime incremental re-opt.
            if ctx.cycle_guard.contains(&branch_id)
                || arms.iter().any(|a| ctx.cycle_guard.contains(a))
            {
                continue;
            }

            // (1) Duplicate-path convergence: count distinct routes.
            let distinct = Self::distinct_arm_count(graph, &arms);

            // (2) Invariant: never strand the last path — a branch always
            //     keeps ≥1 viable `(device, backend)` route.
            debug_assert!(
                distinct >= 1,
                "FrontierConvergence: branch {branch_id:?} would strand to zero routes",
            );

            // The shipped PR-A4 pathfinder emits arm-1 on a distinct
            // backend from arm-0, so every arm is a distinct route and no
            // merge is needed. Assert that — if a future pathfinder leaks
            // a forward-identical arm, this fires and signals that an
            // in-place arm-rewrite (a `fuel-graph` setter) must land with
            // it. We deliberately do NOT silently rewrite arms here in B3
            // (it would need a new `fuel-graph` mutation API and is a
            // behavior change for a case that cannot arise from the
            // shipped passes) — B3 is a structural refactor only.
            debug_assert_eq!(
                distinct,
                arms.len(),
                "FrontierConvergence: branch {branch_id:?} has forward-identical arms \
                 ({} distinct of {}); a multi-route pathfinder leaked a duplicate — \
                 land an in-graph arm-merge with it",
                distinct,
                arms.len(),
            );
        }

        Ok(())
    }
}

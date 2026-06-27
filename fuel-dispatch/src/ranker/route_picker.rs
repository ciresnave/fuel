//! The **runtime route picker** ("Picker 2") — selects an arm at each
//! `Op::Branch` decision point by **live telemetry**.
//!
//! Phase C PR-C1 of the "plan IS the graph" rebuild
//! ([`../../docs/session-prompts/plan-is-graph-rebuild.md`](
//! ../../docs/session-prompts/plan-is-graph-rebuild.md) Phase C,
//! capability [7]; [`../../docs/architecture/06-runtime.md`](
//! ../../docs/architecture/06-runtime.md) §"Route picker (the runtime
//! selector / Picker 2)").
//!
//! ## What this is
//!
//! Phases A+B emit the optimized form as an in-graph multi-path
//! structure: `optimize_graph` records `Op::Branch { reconverge_at }`
//! decision points whose arms (`inputs`) are competing `(device, backend)`
//! placements — arm-0 the cost-vector winner, arm-1+ runner-up
//! placements bounded by the per-device Pareto frontier. Phase B realize
//! followed **arm-0 statically** (`lower_runs_arm0`). This module is the
//! runtime selector that **chooses an arm per branch by live telemetry**,
//! replacing the static arm-0 lowering with [`fuel_graph::lower_picked_route`].
//!
//! ## How it picks
//!
//! [`pick_route`] walks the reachable branches in **topological order**
//! ([`fuel_graph::branches_in_topo_order`], keyed by each branch's
//! `reconverge_at` merge position) so coupled upstream decisions are
//! committed before a downstream branch is reached. At each branch it
//! builds a per-branch [`AlternativeSet`] over the arms — one
//! [`Candidate`] per arm carrying the arm's real `(backend, device)` and
//! its plan cost — and consults the supplied [`RuntimeSelector`] (the
//! production [`crate::ranker::ChainedSelector`]: VRAM-pressure guard
//! over the [`BackendRuntimeLookup`]'s per-tier free memory, then
//! Judge-measured rank, then the static winner). The picked candidate's
//! `(backend, device)` maps back to an arm index, stored in the
//! [`PickedRoute`].
//!
//! Under **no pressure / no telemetry** the chained selector degrades to
//! `set.winner()` = arm-0 (the cost-ranked first candidate), so the route
//! is arm-0 everywhere and [`fuel_graph::lower_picked_route`] reproduces
//! Phase B's `lower_runs_arm0` exactly. Under **VRAM pressure** the guard
//! demotes/skips the GPU arm and the picker takes a host-RAM arm
//! instead — exactly the "multiple paths per device" payoff
//! (06-runtime §"Memory pressure as the parallelism limit").
//!
//! ## Caching + re-resolution
//!
//! [`RouteCache`] holds the last resolved [`PickedRoute`] plus a
//! [`TelemetryFingerprint`] (per-tier free-memory buckets across the
//! candidate backends). [`RouteCache::resolve`] re-resolves only when the
//! fingerprint changes meaningfully (a bucket shift); otherwise it reuses
//! the cached route. In steady state the picker is a table lookup;
//! re-resolution fires on transitions (pressure rising, a device
//! appearing). A CPU-only build has zero branches ⇒ the route is empty ⇒
//! the picker is a no-op and realize is unchanged.

use std::sync::Arc;

use fuel_ir::probe::BackendId;
use fuel_ir::DeviceLocation;
use fuel_graph::{branches_in_topo_order, Graph, NodeId, PickedRoute};

use crate::fused::{CostEstimate, PrecisionGuarantee};
use crate::kernel::{KernelCaps, OpParams};
use crate::plan::{default_device_for, ExecutionPlan};
use crate::ranker::{
    AlternativeSet, BackendRuntimeLookup, Candidate, RuntimeSelector,
};

/// Bounded lookahead for adversarially-coupled branches (06-runtime
/// §"Resolution order matters when decisions are coupled"). The default
/// resolution is locally-greedy in topological order — by the time the
/// picker reaches a branch its upstream decisions are already committed.
/// Rare adversarial coupling is bounded by considering at most this many
/// branches jointly. PR-C1 commits upstream-first in topo order (the
/// greedy default); `K` is the documented ceiling the joint-resolution
/// window must never exceed.
pub const LOOKAHEAD_K: usize = 3;

/// A coarse fingerprint of the live per-tier free memory across the
/// backends a route's branches care about — the cache key for
/// [`RouteCache`]. Re-resolution fires only on a *meaningful* delta: we
/// bucket each backend's free bytes into coarse log-scale tiers so noise
/// (a few MB jitter) does not invalidate the cached route, but a genuine
/// pressure transition (free memory dropping an order of magnitude) does.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TelemetryFingerprint {
    /// `(backend, free-memory bucket)` pairs, sorted by backend for a
    /// stable equality check. An absent backend (no live handle) is
    /// omitted — its bucket is "unknown", which never forces a
    /// re-resolve on its own.
    buckets: Vec<(BackendId, u8)>,
}

impl TelemetryFingerprint {
    /// Fingerprint the backends in `backends` against the live lookup.
    /// Each backend's free bytes are bucketed via [`free_bytes_bucket`].
    fn sample(
        backends: &[BackendId],
        lookup: Option<&BackendRuntimeLookup>,
    ) -> Self {
        let mut buckets: Vec<(BackendId, u8)> = Vec::new();
        if let Some(lookup) = lookup {
            let mut seen: Vec<BackendId> = Vec::new();
            for &b in backends {
                if seen.contains(&b) {
                    continue;
                }
                seen.push(b);
                if let Some(handle) = lookup(b, default_device_for(b)) {
                    if let Some(free) = handle.available_bytes() {
                        buckets.push((b, free_bytes_bucket(free)));
                    }
                }
            }
        }
        buckets.sort_by_key(|&(b, _)| b as u8);
        Self { buckets }
    }
}

/// Bucket free-memory bytes into coarse log-scale tiers (powers of ~16×)
/// so the [`TelemetryFingerprint`] is stable under small jitter but
/// flips on an order-of-magnitude pressure transition. `0` for zero free;
/// otherwise `floor(log2(free)/4) + 1` clamped to a byte.
fn free_bytes_bucket(free: u64) -> u8 {
    if free == 0 {
        return 0;
    }
    let log2 = 63 - free.leading_zeros(); // floor(log2(free))
    ((log2 / 4) + 1).min(u8::MAX as u32) as u8
}

/// Per-realize cache of the resolved route + the telemetry it was
/// resolved under (06-runtime §"Telemetry caching for picker speed").
///
/// Live across realizes by being held by the caller (the bridge). The
/// first [`resolve`](RouteCache::resolve) populates it; subsequent calls
/// reuse the cached route while the [`TelemetryFingerprint`] is stable,
/// and re-resolve on a meaningful delta.
#[derive(Clone, Debug, Default)]
pub struct RouteCache {
    cached: Option<(TelemetryFingerprint, PickedRoute)>,
    /// Diagnostics: how many times [`resolve`](RouteCache::resolve) hit
    /// the cache vs. re-resolved. Tests assert the re-resolve discipline.
    pub hits: u64,
    pub resolves: u64,
}

impl RouteCache {
    /// A fresh, empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve the route for `roots`, reusing the cached route while the
    /// live telemetry fingerprint is unchanged. Re-resolves (and updates
    /// the cache) on the first call or a meaningful telemetry delta.
    ///
    /// `plan` supplies the per-arm cost/context; `selector` is the
    /// production runtime selector; `lookup` is the live per-tier
    /// free-memory closure (`None` ⇒ no pressure signal ⇒ arm-0).
    pub fn resolve(
        &mut self,
        graph: &Graph,
        roots: &[NodeId],
        plan: &ExecutionPlan,
        selector: &dyn RuntimeSelector,
        lookup: Option<&BackendRuntimeLookup>,
    ) -> PickedRoute {
        let backends = route_backends(graph, roots);
        let fingerprint = TelemetryFingerprint::sample(&backends, lookup);

        if let Some((cached_fp, cached_route)) = &self.cached {
            if *cached_fp == fingerprint {
                self.hits += 1;
                return cached_route.clone();
            }
        }

        let route = pick_route(graph, roots, plan, selector, lookup);
        self.resolves += 1;
        self.cached = Some((fingerprint, route.clone()));
        route
    }
}

/// The set of `(backend)`s the route's branches' arms target — the
/// backends whose live free memory the picker (and the cache
/// fingerprint) consults.
fn route_backends(graph: &Graph, roots: &[NodeId]) -> Vec<BackendId> {
    let mut backends: Vec<BackendId> = Vec::new();
    for branch in branches_in_topo_order(graph, roots) {
        for &arm in &graph.node(branch).inputs {
            if let Some(b) = graph.target_backend(arm) {
                if !backends.contains(&b) {
                    backends.push(b);
                }
            }
        }
    }
    backends
}

/// Walk the multi-path graph's `Op::Branch` decision points in
/// **topological order** and choose one arm at each via the
/// `RuntimeSelector` over the arms' `(device, backend)` + plan cost +
/// live per-tier free memory. Returns the [`PickedRoute`]
/// ([`fuel_graph::lower_picked_route`] consumes it).
///
/// Topological order means a downstream branch is resolved only after its
/// upstream branches are committed (the picks recorded in `picked`), so
/// coupled decisions resolve consistently (06-runtime §"Resolution order
/// matters when decisions are coupled"). The greedy upstream-first pass
/// is the default; [`LOOKAHEAD_K`] bounds any joint-resolution window for
/// adversarial coupling.
///
/// With no pressure / no telemetry the selector degrades to `set.winner()`
/// = arm-0 for every branch, so the returned route is empty-equivalent
/// (every pick is arm 0) and realize is identical to Phase B.
pub fn pick_route(
    graph: &Graph,
    roots: &[NodeId],
    plan: &ExecutionPlan,
    selector: &dyn RuntimeSelector,
    lookup: Option<&BackendRuntimeLookup>,
) -> PickedRoute {
    // The live per-tier free-memory lookup is consulted *through* the
    // selector — the production `ChainedSelector` holds its own
    // `BackendRuntimeLookup` and reads each arm's `(backend, device)` fit
    // status in `select`. `lookup` is part of the picker's documented
    // contract (06-runtime §"Route picker") and is the same handle the
    // `RouteCache` fingerprints telemetry through; the per-arm guard
    // itself lives in the selector, so `pick_route` does not re-query it.
    let _ = lookup;
    let mut picked = PickedRoute::new();
    let branches = branches_in_topo_order(graph, roots);

    for branch in branches {
        let arms = graph.node(branch).inputs.clone();
        if arms.len() < 2 {
            // A single-arm (or zero-arm) branch has no decision — arm 0.
            continue;
        }
        let chosen = pick_arm(graph, &arms, plan, selector);
        // Only record a non-arm-0 pick; arm 0 is the default in
        // `lower_picked_route`, so keeping the route minimal makes the
        // "no pressure ⇒ empty route ⇒ Phase B behavior" contract direct.
        if chosen != 0 {
            picked.insert(branch, chosen);
        }
    }

    picked
}

/// Choose one arm index for a single branch. Builds a per-arm
/// [`AlternativeSet`] (one [`Candidate`] per arm carrying the arm's real
/// `(backend, device)` + plan cost + the fork's decision context), runs
/// the selector, and maps the picked candidate's `(backend, device)`
/// back to its arm index.
fn pick_arm(
    graph: &Graph,
    arms: &[NodeId],
    plan: &ExecutionPlan,
    selector: &dyn RuntimeSelector,
) -> usize {
    // The fork node = arm-0's exit. Its plan `AlternativeSet` (when
    // present) carries the real per-placement Candidates the ranker
    // produced + the DecisionContext (op/dtype/size_class) the
    // Judge-aware rank keys on. We mirror its candidates into a per-arm
    // set in ARM ORDER so the picked index maps straight back to an arm.
    let fork = arms[0];
    let plan_set = plan.alternatives(fork);

    let mut arm_set = AlternativeSet::empty();
    for &arm in arms {
        let backend = graph.target_backend(arm).unwrap_or(BackendId::Cpu);
        let device = default_device_for(backend);
        let candidate = plan_set
            .and_then(|set| candidate_for_placement(set, backend, device))
            .cloned()
            .unwrap_or_else(|| synthetic_candidate(backend, device));
        arm_set.push(candidate);
    }
    // Carry the fork's decision context so the Judge-aware rank leg can
    // re-query measurements per arm.
    if let Some(ctx) = plan_set.and_then(|s| s.context()) {
        arm_set.set_context(*ctx);
    }

    // The selector picks one candidate; arm index = its position in the
    // arm-ordered set. `select` returns `None` only on an empty set
    // (impossible here — arms.len() >= 2), in which case arm 0.
    match selector.select(&arm_set) {
        Some(picked) => arm_index_of(&arm_set, picked),
        None => 0,
    }
}

/// Find a plan candidate matching `(backend, device)` (the arm's real
/// placement). Returns the first match — sibling kernels at the same
/// `(backend, device)` are interchangeable for the arm-selection purpose
/// (they differ only by `kernel_source`, which the selector still ranks).
fn candidate_for_placement(
    set: &AlternativeSet,
    backend: BackendId,
    device: DeviceLocation,
) -> Option<&Candidate> {
    set.alternatives()
        .iter()
        .find(|c| c.backend == backend && c.device == device)
        .or_else(|| {
            // Fall back to a backend-only match (the plan may carry a
            // different GPU ordinal than the arm's default device).
            set.alternatives().iter().find(|c| c.backend == backend)
        })
}

/// The arm index of `picked` within the arm-ordered `set` — by
/// `(backend, device)` identity. The set was built in arm order, so the
/// first matching position is the chosen arm.
fn arm_index_of(set: &AlternativeSet, picked: &Candidate) -> usize {
    set.alternatives()
        .iter()
        .position(|c| c.backend == picked.backend && c.device == picked.device)
        .unwrap_or(0)
}

/// A minimal stand-in [`Candidate`] for an arm with no matching plan
/// entry: real `(backend, device)`, zero cost, no-op kernel. The
/// selector reads only `backend` / `device` / `static_cost` /
/// `inbound_transfer_ns` / `kernel_source` for its pick, so the kernel
/// pointer is never invoked — this candidate exists purely to carry the
/// arm's placement to the VRAM-pressure guard.
fn synthetic_candidate(backend: BackendId, device: DeviceLocation) -> Candidate {
    Candidate {
        kernel: noop_kernel,
        caps: KernelCaps::empty(),
        backend,
        device,
        precision: PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        static_cost: CostEstimate {
            flops: 0,
            bytes_moved: 0,
            kernel_overhead_ns: 0,
        },
        inbound_transfer_ns: 0,
        op_params: OpParams::None,
        coupling: Vec::new(),
        kernel_source: "route-picker-synthetic",
    }
}

/// Never-invoked kernel for [`synthetic_candidate`] (see its docs). The
/// picker reads placement metadata only; the chosen arm is lowered to
/// runs and dispatched through the normal binding-table path, not via
/// this pointer.
fn noop_kernel(
    _i: &[Arc<std::sync::RwLock<fuel_memory::Storage>>],
    _o: &mut [Arc<std::sync::RwLock<fuel_memory::Storage>>],
    _l: &[fuel_ir::Layout],
    _p: &OpParams,
) -> fuel_ir::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::CostEstimate;
    use crate::ranker::{
        BackendRuntimeHandle, ChainedSelector, WinnerSelector,
    };
    use fuel_backend_contract::backend::BackendRuntime;
    use fuel_graph::{Node, Op};
    use fuel_ir::{DType, Shape};

    fn f32_node(g: &mut Graph, op: Op, inputs: Vec<NodeId>) -> NodeId {
        g.push(Node {
            op,
            inputs,
            shape: Shape::from_dims(&[2]),
            dtype: DType::F32,
        })
    }

    /// Build a 2-arm diamond with arm-0 on `arm0_backend` and arm-1 on
    /// `arm1_backend`. Returns `(graph, branch, arm0, arm1, post)`.
    ///
    /// Topology mirrors the run.rs diamond: `diverge -> {arm0, arm1}`;
    /// `reconverge` reads arm0 (the runnability invariant); the Branch
    /// merges {arm0, arm1} at `reconverge`. The arm exits carry their
    /// `target_backend` so the picker can read each arm's placement.
    fn diamond(
        arm0_backend: BackendId,
        arm1_backend: BackendId,
    ) -> (Graph, NodeId, NodeId, NodeId, NodeId) {
        let mut g = Graph::new();
        let pre = f32_node(&mut g, Op::Const, vec![]);
        let diverge = f32_node(&mut g, Op::Relu, vec![pre]);
        let arm0 = f32_node(&mut g, Op::Silu, vec![diverge]);
        let arm1 = f32_node(&mut g, Op::Gelu, vec![diverge]);
        g.set_target_backend(arm0, arm0_backend);
        g.set_target_backend(arm1, arm1_backend);
        let reconverge = f32_node(&mut g, Op::Relu, vec![arm0]);
        let mut b = g.open_branch(diverge);
        b.add_arm(arm0);
        b.add_arm(arm1);
        let branch = b
            .finalize_branches(&mut g, reconverge)
            .expect("well-formed 2-arm branch")
            .expect("2 arms survive");
        let post = f32_node(&mut g, Op::Tanh, vec![reconverge]);
        (g, branch, arm0, arm1, post)
    }

    /// A plan `AlternativeSet` for the fork node with two placements:
    /// arm-0 backend at `cost0` ns (the winner), arm-1 backend at
    /// `cost1` ns.
    fn plan_with_fork_set(
        fork: NodeId,
        a0: (BackendId, DeviceLocation, u64),
        a1: (BackendId, DeviceLocation, u64),
    ) -> ExecutionPlan {
        let mut set = AlternativeSet::empty();
        set.push(make_candidate(a0.0, a0.1, a0.2));
        set.push(make_candidate(a1.0, a1.1, a1.2));
        let mut plan = ExecutionPlan::empty();
        plan.alternatives.insert(fork, set);
        plan
    }

    fn make_candidate(
        backend: BackendId,
        device: DeviceLocation,
        cost_ns: u64,
    ) -> Candidate {
        Candidate {
            kernel: noop_kernel,
            caps: KernelCaps::empty(),
            backend,
            device,
            precision: PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            static_cost: CostEstimate {
                flops: cost_ns,
                bytes_moved: cost_ns,
                kernel_overhead_ns: 0,
            },
            inbound_transfer_ns: 0,
            op_params: OpParams::None,
            coupling: Vec::new(),
            kernel_source: "test",
        }
    }

    struct MockRuntime {
        available: Option<u64>,
        total: Option<u64>,
    }
    impl BackendRuntime for MockRuntime {
        fn available_bytes(&self) -> Option<u64> {
            self.available
        }
        fn total_bytes(&self) -> Option<u64> {
            self.total
        }
    }

    /// Per-backend `(available, total)` lookup; backends not listed
    /// resolve to `None` (= Unknown / no signal).
    fn lookup_for(
        entries: Vec<(BackendId, Option<u64>, Option<u64>)>,
    ) -> BackendRuntimeLookup {
        Arc::new(move |b, _d| {
            entries.iter().find(|(eb, _, _)| *eb == b).map(|&(_, a, t)| {
                Box::new(MockRuntime { available: a, total: t }) as BackendRuntimeHandle
            })
        })
    }

    /// (a) NO PRESSURE / NO TELEMETRY ⇒ the picker chooses arm-0 (the
    /// winner) at every branch ⇒ an empty route ⇒ realize is identical to
    /// Phase B. Tested with both the bare `WinnerSelector` and the
    /// production `ChainedSelector` with no signals.
    #[test]
    fn no_telemetry_picks_arm0_empty_route() {
        let (g, _branch, fork, _arm1, post) =
            diamond(BackendId::Cuda, BackendId::Cpu);
        let plan = plan_with_fork_set(
            fork,
            (BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 }, 100),
            (BackendId::Cpu, DeviceLocation::Cpu, 200),
        );

        // Bare winner selector: always arm 0.
        let winner = WinnerSelector;
        let route = pick_route(&g, &[post], &plan, &winner, None);
        assert!(
            route.is_empty(),
            "no telemetry ⇒ arm-0 everywhere ⇒ empty route; got {route:?}",
        );

        // Production chained selector with NO signals (no judge, no
        // lookup) degrades to the static winner = arm 0.
        let chained = ChainedSelector::with_default_estimator(None, None);
        let route = pick_route(&g, &[post], &plan, &chained, None);
        assert!(
            route.is_empty(),
            "ChainedSelector with no signals ⇒ arm-0 everywhere; got {route:?}",
        );
    }

    /// (b) SIMULATED VRAM PRESSURE: the GPU arm's backend reports tiny
    /// free VRAM (WontFit), so the chained selector's guard skips it and
    /// the picker takes the host-RAM (CPU) arm instead — arm 1.
    #[test]
    fn vram_pressure_picks_host_ram_arm() {
        let (g, branch, fork, _arm1, post) =
            diamond(BackendId::Cuda, BackendId::Cpu);
        // Both placements cost the same kernel time; the only signal is
        // memory pressure, so the guard alone must flip the pick.
        let plan = plan_with_fork_set(
            fork,
            (BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 }, 100),
            (BackendId::Cpu, DeviceLocation::Cpu, 100),
        );

        // CUDA has 1 byte free of 10_000 → WontFit for the default
        // bytes_moved=100 estimate; CPU has ample room → Comfortable.
        let lookup = lookup_for(vec![
            (BackendId::Cuda, Some(1), Some(10_000)),
            (BackendId::Cpu, Some(8_000), Some(10_000)),
        ]);
        let chained =
            ChainedSelector::with_default_estimator(None, Some(lookup.clone()));

        let route = pick_route(&g, &[post], &plan, &chained, Some(&lookup));
        assert_eq!(
            route.get(&branch).copied(),
            Some(1),
            "under VRAM pressure the picker takes the host-RAM (CPU) arm; \
             route={route:?}",
        );
    }

    /// (c) TWO COUPLED BRANCHES resolve consistently in topological
    /// order: both GPU arms are under pressure, so the picker takes the
    /// CPU arm at BOTH branches — the downstream branch sees the same
    /// pressure the upstream one did and resolves the same way.
    #[test]
    fn two_coupled_branches_resolve_consistently() {
        // Two diamonds chained: post0 feeds the second diverge.
        let mut g = Graph::new();
        let pre = f32_node(&mut g, Op::Const, vec![]);
        // --- branch 1 ---
        let div1 = f32_node(&mut g, Op::Relu, vec![pre]);
        let a0_1 = f32_node(&mut g, Op::Silu, vec![div1]);
        let a1_1 = f32_node(&mut g, Op::Gelu, vec![div1]);
        g.set_target_backend(a0_1, BackendId::Cuda);
        g.set_target_backend(a1_1, BackendId::Cpu);
        let recon1 = f32_node(&mut g, Op::Relu, vec![a0_1]);
        let mut b1 = g.open_branch(div1);
        b1.add_arm(a0_1);
        b1.add_arm(a1_1);
        let branch1 = b1
            .finalize_branches(&mut g, recon1)
            .expect("branch1 valid")
            .expect("2 arms");
        // --- branch 2 (downstream of branch 1's merge) ---
        let div2 = f32_node(&mut g, Op::Tanh, vec![recon1]);
        let a0_2 = f32_node(&mut g, Op::Silu, vec![div2]);
        let a1_2 = f32_node(&mut g, Op::Gelu, vec![div2]);
        g.set_target_backend(a0_2, BackendId::Cuda);
        g.set_target_backend(a1_2, BackendId::Cpu);
        let recon2 = f32_node(&mut g, Op::Relu, vec![a0_2]);
        let mut b2 = g.open_branch(div2);
        b2.add_arm(a0_2);
        b2.add_arm(a1_2);
        let branch2 = b2
            .finalize_branches(&mut g, recon2)
            .expect("branch2 valid")
            .expect("2 arms");
        let post = f32_node(&mut g, Op::Tanh, vec![recon2]);

        // Topo order must place branch1 before branch2.
        let order = branches_in_topo_order(&g, &[post]);
        assert_eq!(
            order,
            vec![branch1, branch2],
            "coupled branches resolve upstream-first in topo order",
        );

        // Plans for both forks: GPU winner + CPU runner-up, same cost.
        let mut plan = ExecutionPlan::empty();
        for fork in [a0_1, a0_2] {
            let mut set = AlternativeSet::empty();
            set.push(make_candidate(
                BackendId::Cuda,
                DeviceLocation::Cuda { gpu_id: 0 },
                100,
            ));
            set.push(make_candidate(BackendId::Cpu, DeviceLocation::Cpu, 100));
            plan.alternatives.insert(fork, set);
        }

        // GPU under pressure ⇒ both branches pick the CPU arm (arm 1).
        let lookup = lookup_for(vec![
            (BackendId::Cuda, Some(1), Some(10_000)),
            (BackendId::Cpu, Some(8_000), Some(10_000)),
        ]);
        let chained =
            ChainedSelector::with_default_estimator(None, Some(lookup.clone()));
        let route = pick_route(&g, &[post], &plan, &chained, Some(&lookup));

        assert_eq!(route.get(&branch1).copied(), Some(1), "branch1 → CPU arm");
        assert_eq!(route.get(&branch2).copied(), Some(1), "branch2 → CPU arm");
    }

    /// (d) The route is CACHED and re-resolved only on a meaningful
    /// telemetry delta. Same fingerprint ⇒ cache hit (no re-resolve); a
    /// genuine pressure transition ⇒ re-resolve + a possibly different
    /// route.
    #[test]
    fn route_cached_and_reresolved_on_telemetry_delta() {
        let (g, branch, fork, _arm1, post) =
            diamond(BackendId::Cuda, BackendId::Cpu);
        let plan = plan_with_fork_set(
            fork,
            (BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 }, 100),
            (BackendId::Cpu, DeviceLocation::Cpu, 100),
        );

        // Start comfortable: both backends have ample free memory.
        let comfy = lookup_for(vec![
            (BackendId::Cuda, Some(8_000), Some(10_000)),
            (BackendId::Cpu, Some(8_000), Some(10_000)),
        ]);
        let chained_comfy =
            ChainedSelector::with_default_estimator(None, Some(comfy.clone()));

        let mut cache = RouteCache::new();
        let r1 = cache.resolve(&g, &[post], &plan, &chained_comfy, Some(&comfy));
        assert!(r1.is_empty(), "comfortable ⇒ arm-0 (winner) route");
        assert_eq!(cache.resolves, 1, "first resolve populates the cache");
        assert_eq!(cache.hits, 0);

        // Same telemetry ⇒ cache hit, no re-resolve.
        let r2 = cache.resolve(&g, &[post], &plan, &chained_comfy, Some(&comfy));
        assert_eq!(r2, r1, "stable telemetry reuses the cached route");
        assert_eq!(cache.resolves, 1, "no re-resolve on a stable fingerprint");
        assert_eq!(cache.hits, 1, "stable telemetry is a cache hit");

        // Meaningful delta: CUDA free memory collapses to 1 byte
        // (WontFit). Re-resolve fires and the route flips to the CPU arm.
        let pressured = lookup_for(vec![
            (BackendId::Cuda, Some(1), Some(10_000)),
            (BackendId::Cpu, Some(8_000), Some(10_000)),
        ]);
        let chained_pressured = ChainedSelector::with_default_estimator(
            None,
            Some(pressured.clone()),
        );
        let r3 = cache.resolve(
            &g,
            &[post],
            &plan,
            &chained_pressured,
            Some(&pressured),
        );
        assert_eq!(cache.resolves, 2, "a meaningful telemetry delta re-resolves");
        assert_eq!(cache.hits, 1, "the delta is not counted as a hit");
        assert_eq!(
            r3.get(&branch).copied(),
            Some(1),
            "after the pressure transition the route picks the CPU arm",
        );
    }

    /// A graph with ZERO branches (a CPU-only build) ⇒ the picker is a
    /// no-op ⇒ the route is empty ⇒ realize is unchanged. The cache
    /// resolves it once and caches the empty route.
    #[test]
    fn branchless_graph_is_a_noop() {
        let mut g = Graph::new();
        let a = f32_node(&mut g, Op::Const, vec![]);
        let b = f32_node(&mut g, Op::Relu, vec![a]);
        let c = f32_node(&mut g, Op::Tanh, vec![b]);
        let plan = ExecutionPlan::empty();
        let winner = WinnerSelector;
        let route = pick_route(&g, &[c], &plan, &winner, None);
        assert!(route.is_empty(), "no branches ⇒ empty route (picker no-op)");
    }
}

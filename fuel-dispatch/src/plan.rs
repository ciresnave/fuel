//! Execution plan — per-realize compilation output, Phase 1.5
//! reshape of the picker-work arc.
//!
//! The plan is a topological order plus a sparse `NodeId ->
//! AlternativeSet` map. Each entry is the per-decision-point top-N
//! candidates the optimizer ranker considered, filtered and ranked
//! per the configured `PlanOptions`. The eventual executor reads
//! `alternatives` per node, takes the winner (Picker 2 territory in
//! the long-term shape; Phase 4 wires the executor to this surface),
//! and dispatches the resolved kernel.
//!
//! ## What changed in Phase 1.5
//!
//! Pre-1.5 this module hosted `NodeKernelBinding`, `TolerancePolicy`,
//! and `resolve_kernel` — the v1 picker that fell out of Phase 7.6
//! step 9b. Those types had zero callers (verified by the
//! 2026-05-30 picker-alternatives audit), so Phase 1.5 retires
//! them and ships the AlternativeSet-based plan in their place. The
//! shape composes with everything Phase 1.1–1.4 already shipped:
//! the enumerator builds candidates, the filter chain narrows them,
//! the cost composer scores them, and the plan stores the
//! top-N-after-filter-and-rank.
//!
//! Phase 4 of the picker-work arc migrates the PipelinedExecutor to
//! consume `ExecutionPlan::alternatives` instead of `compile_node`'s
//! first-registered path. Until then this module ships the
//! infrastructure with no production consumer.

use std::collections::HashMap;

use fuel_ir::dispatch::{OpKind, SizeClass};
use fuel_ir::probe::BackendId;
use fuel_ir::{DType, DeviceLocation, Error, Result, Shape};
use fuel_graph::{Graph, NodeId};

use smallvec::SmallVec;

use crate::kernel::KernelBindingTable;
use crate::pipelined::{build_lookup_dtypes, op_to_op_kind};
use crate::ranker::{
    apply_filter_chain, apply_inbound_transfer_costs, composite_ns,
    compute_static_costs, default_chain, enumerate_candidates, AlternativeSet,
    CapabilitiesLookup, ChainInput, DecisionContext, FilterContext, JudgeOracle,
    PlacementDp, PrecisionRequirement, TransferEstimator, KEEP_PER_DEVICE,
};

/// Per-realize execution plan. Built by [`compile_plan`].
///
/// `alternatives` is sparse — view ops, `Op::Const`, and ops
/// `op_to_op_kind` returns `None` for get no entry. The executor's
/// per-node dispatch reads `alternatives.get(&id)` and either takes
/// the winner (top entry after rank) or, in the long-term shape,
/// hands the full set to the runtime selector (Picker 2) for
/// layer-3-telemetry-driven choice.
#[derive(Debug)]
pub struct ExecutionPlan {
    /// Topological order — same shape the executor walks.
    pub order: Vec<NodeId>,
    /// One `AlternativeSet` per kernel-bearing node. Sparse.
    pub alternatives: HashMap<NodeId, AlternativeSet>,
    /// `SystemTopology` generation counter snapshotted at
    /// [`compile_plan`] time. The executor (Phase 4.3) checks this
    /// against `crate::dispatch::topology_generation()` at every
    /// dispatch-chunk boundary; mismatch surfaces
    /// [`Error::TopologyChanged`] so the realize layer can rebuild
    /// the plan against the fresh topology.
    ///
    /// `0` for [`ExecutionPlan::empty()`] — empty plans by definition
    /// have no alternatives to invalidate, so the executor's chunk-
    /// boundary check skips them.
    pub generation: u64,
}

impl ExecutionPlan {
    /// Empty plan. Tests + future direct-call sites.
    pub fn empty() -> Self {
        Self {
            order: Vec::new(),
            alternatives: HashMap::new(),
            generation: 0,
        }
    }

    /// Read-only handle to a node's alternative set. `None` for
    /// nodes outside the plan's map.
    pub fn alternatives(&self, node: NodeId) -> Option<&AlternativeSet> {
        self.alternatives.get(&node)
    }

    /// Mutable handle — used by post-plan refinement passes (Phase
    /// 3's Judge integration may re-rank top-N after the plan is
    /// built).
    pub fn alternatives_mut(&mut self, node: NodeId) -> Option<&mut AlternativeSet> {
        self.alternatives.get_mut(&node)
    }
}

/// Default [`DeviceLocation`] for a `BackendId` when the graph has
/// no per-node placement set. Mirrors the convention in
/// `fuel-graph-router` (CPU is the singleton `Cpu`; GPU backends
/// default to ordinal 0).
pub fn default_device_for(backend: BackendId) -> DeviceLocation {
    match backend {
        BackendId::Cpu => DeviceLocation::Cpu,
        BackendId::Cuda => DeviceLocation::Cuda { gpu_id: 0 },
        BackendId::Vulkan => DeviceLocation::Vulkan { gpu_id: 0 },
        BackendId::Metal => DeviceLocation::Metal { gpu_id: 0 },
        _ => DeviceLocation::Cpu,
    }
}

/// The `BackendId` that owns a `DeviceLocation`'s storage substrate.
/// Total — `BackendId` mirrors `DeviceLocation` 1:1 (AOCL/MKL are
/// `kernel_source` siblings under `Cpu`, not distinct backends).
/// Inverse of [`default_device_for`] modulo GPU ordinals.
pub fn backend_for_device(loc: DeviceLocation) -> BackendId {
    match loc {
        DeviceLocation::Cpu => BackendId::Cpu,
        DeviceLocation::Cuda { .. } => BackendId::Cuda,
        DeviceLocation::Vulkan { .. } => BackendId::Vulkan,
        DeviceLocation::Metal { .. } => BackendId::Metal,
    }
}

/// Plan-time configuration the caller hands to [`compile_plan`].
///
/// Most fields have sensible defaults via [`PlanOptions::new`] +
/// chained builders. The two callback fields are required for
/// non-trivial usage — the `'env` lifetime ties them to the
/// caller's environment.
///
/// # `placements_for_device`
///
/// When `Some`, called per kernel-bearing node with the node's
/// target `DeviceLocation` and expected to return every
/// `BackendId` the picker is allowed to consider at that device.
/// This is the cross-co-located-backend integration point —
/// callers wire it to `SystemTopology::backends_for(device)` to
/// unlock the AOCL/MKL/CPU competition story.
///
/// When `None`, the planner uses single-backend placement: only
/// `(target_backend, placement)` from the graph node's side-table.
/// Matches the pre-1.5 picker's pinned-backend behavior; useful as
/// a transitional default and for tests.
///
/// # `pinned_device`
///
/// Picker-arc step 4a. When `Some`, nodes WITHOUT an explicit graph
/// placement resolve their decision device to this location — the
/// realize call's pinned DEVICE. This replaces the bridge's
/// pre-plan monolithic per-node `set_target_backend` loop: the
/// graph no longer needs `target_backend` stamped before planning;
/// the plan's per-node winner is stamped back onto the graph AFTER
/// planning (`fuel-core::pipelined_bridge::stamp_plan_backends`).
///
/// Resolution priority per node: explicit `Graph::placement` (the
/// scheduler's per-node assignment) → `pinned_device` →
/// `default_device_for(target_backend)` (legacy stamped graphs) →
/// error.
///
/// # `fallback_placements_for`
///
/// Picker-arc step 4b, relaxed by planner Stage 2. The closure
/// supplies the OFF-DEVICE placements the picker may consider
/// beyond the node's decision device. Two admission regimes:
///
/// - **Priced (Stage 2)** — when a [`Self::transfer_estimator`] is
///   configured (with `populate_costs` + a capabilities lookup),
///   off-device placements ALWAYS enumerate alongside the decision
///   device's, and every candidate carries an inbound-transfer
///   term; locality emerges from the pricing (ties break toward
///   the decision device via the rank's stable sort). After the
///   rank the set is pruned to the winner's device. Nodes with an
///   explicit `Graph::placement` are HARD pins — they stay on the
///   legacy regime below.
/// - **Legacy (unpriced)** — without an estimator, the closure is
///   consulted only when the decision device can't deliver: empty
///   enumeration for the `(op, dtypes)`, or (Stage-2 fix) a hard
///   filter chain that rejects every decision-device registration.
///
/// Constraints enforced by `compile_plan` in both regimes:
///
/// - **Destructive ops never fall back** (`Op::destructive_input()`
///   is `Some`): in-place mutation semantics don't survive moving
///   the op away from the device that owns its mutation target;
///   those ops keep the plan-time `NoBackendForOp` error.
/// - **The surviving set lives on ONE device** (legacy fallback
///   sets freeze to their single ranked winner; relaxed sets prune
///   to the winner's device): the residency stitch (`Op::Copy`
///   insertion) is a graph rewrite computed from the static
///   winner, so a dispatch-time selector must not be able to pick
///   a sibling on a different device.
///
/// # `capabilities_for`
///
/// Required when `populate_costs` is true (the default). The
/// closure resolves a `BackendId` to its `BackendCapabilities` —
/// callers typically wire it to `SystemTopology::capabilities(...)`
/// or the CapabilityRegistry. Returns `None` for backends not in
/// the topology (their candidates retain default zero cost).
pub struct PlanOptions<'env> {
    /// Hard precision floor the picker enforces. Default: empty
    /// (unconstrained — every candidate passes the precision filter).
    pub precision_requirement: PrecisionRequirement,
    /// Whether to invoke [`compute_static_costs`] + rank after
    /// filtering. Default: true. Disable for tests that just want
    /// to verify enumeration + filtering without cost machinery.
    pub populate_costs: bool,
    /// Cross-co-located-backend placement enumerator. See struct
    /// docs.
    pub placements_for_device:
        Option<&'env (dyn Fn(DeviceLocation) -> Vec<BackendId> + 'env)>,
    /// Realize-call device pin (picker-arc step 4a). See struct
    /// docs.
    pub pinned_device: Option<DeviceLocation>,
    /// Off-device fallback enumerator for missing-impl ops
    /// (picker-arc step 4b). See struct docs.
    pub fallback_placements_for: Option<
        &'env (dyn Fn(DeviceLocation) -> Vec<(BackendId, DeviceLocation)> + 'env),
    >,
    /// Capabilities lookup. Required when `populate_costs` is
    /// true.
    pub capabilities_for: Option<&'env CapabilitiesLookup<'env>>,
    /// Optional Phase 3 Layer-2 oracle. When `Some` AND
    /// `populate_costs` is true, the cost composer refines each
    /// candidate's Layer-1 estimate with the Judge's measured
    /// latency at that cell. Candidates the oracle hasn't measured
    /// keep the Layer-1 estimate (silent fallback — absence ≠ zero).
    pub judge: Option<&'env dyn JudgeOracle>,
    /// Planner Stage-2 transfer-cost oracle. When `Some` AND
    /// `populate_costs` is true (with a capabilities lookup), the
    /// cost composer adds a per-candidate inbound-transfer term:
    /// the sum over the node's inputs of
    /// `estimate_transfer_ns(input residency, candidate device,
    /// input bytes)`, for every input whose residency is known at
    /// plan time. Producer residencies are committed along the topo
    /// walk (producers rank before consumers); graph-input
    /// residencies come from [`Self::input_residency`].
    ///
    /// Also the gate for the Stage-2 admission relax: with pricing
    /// active, off-device candidates ALWAYS enumerate (priced) for
    /// nodes without an explicit `Graph::placement` — locality
    /// emerges from the numbers, not from the legacy
    /// missing-impl-only fallback gate.
    ///
    /// Production callers wire this to
    /// `SystemTopology::estimate_transfer_ns` (Stage-1
    /// `TransferCalibration` behind it); unit tests use synthetic
    /// estimators — never live calibration.
    pub transfer_estimator: Option<&'env dyn TransferEstimator>,
    /// Residency resolver for graph INPUTS whose bytes already
    /// exist outside the plan — `Op::Const` storages uploaded by
    /// the const cache and persistent `initial` slots
    /// (InferenceContext). Consulted while threading residency
    /// through the plan walk, with the same priority the bridge's
    /// `effective_placements` gives cached storages: after
    /// residency-declaring ops and explicit placements, before the
    /// plan's own winner. `None` for a node means "not a resident
    /// input" — its residency resolves through the remaining rules
    /// (or stays unknown, in which case no transfer term fires on
    /// its edges).
    pub input_residency: Option<&'env (dyn Fn(NodeId) -> Option<DeviceLocation> + 'env)>,
    /// Planner Stage 4: incremental extension over a previously
    /// compiled plan. Nodes already covered by `reuse_plan` (an
    /// `AlternativeSet` exists for them) skip enumeration / filter /
    /// cost / DP entirely — their stored set is cloned into the new
    /// plan and their residency commits as the stored winner's
    /// device (mirroring the priority-4 "plan winner" rule), so the
    /// delta's pricing sees the same residency picture the original
    /// walk committed.
    ///
    /// The caller (the plan store) is responsible for only reusing
    /// plans built under the SAME topology generation and pinned
    /// device — `compile_plan` does not re-validate the reused sets.
    /// Reused candidates keep their original `inbound_transfer_ns`
    /// terms; a residency shift between builds is repriced only for
    /// delta nodes (acceptable drift — placement correctness is
    /// re-stitched per realize by the bridge's residency passes).
    pub reuse_plan: Option<&'env ExecutionPlan>,
}

impl Default for PlanOptions<'_> {
    fn default() -> Self {
        Self {
            precision_requirement: PrecisionRequirement::default(),
            populate_costs: true,
            placements_for_device: None,
            pinned_device: None,
            fallback_placements_for: None,
            capabilities_for: None,
            judge: None,
            transfer_estimator: None,
            input_residency: None,
            reuse_plan: None,
        }
    }
}

impl<'env> PlanOptions<'env> {
    /// Builder constructor; same fields as [`Default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the precision requirement hard filter.
    pub fn with_precision_requirement(mut self, req: PrecisionRequirement) -> Self {
        self.precision_requirement = req;
        self
    }

    /// Disable cost computation + ranking. Sets `populate_costs =
    /// false`; alternatives end up in enumerator order, useful for
    /// tests that don't care about ranking.
    pub fn without_cost_population(mut self) -> Self {
        self.populate_costs = false;
        self
    }

    /// Attach a co-located-backend enumerator. Wire to
    /// `SystemTopology::backends_for(device)` for production.
    pub fn with_placements_for_device(
        mut self,
        f: &'env (dyn Fn(DeviceLocation) -> Vec<BackendId> + 'env),
    ) -> Self {
        self.placements_for_device = Some(f);
        self
    }

    /// Pin the realize call's device. Nodes without an explicit
    /// graph placement enumerate their candidates at this location
    /// — no per-node `target_backend` stamp needed before planning.
    /// Picker-arc step 4a.
    pub fn with_pinned_device(mut self, device: DeviceLocation) -> Self {
        self.pinned_device = Some(device);
        self
    }

    /// Attach an off-device fallback enumerator. With a Stage-2
    /// transfer estimator configured, off-device placements always
    /// enumerate, priced; without one, the closure is consulted
    /// only when the decision device can't deliver (no impl, or
    /// every impl hard-filtered). Wire to the `SystemTopology`
    /// device list for production. Picker-arc step 4b + planner
    /// Stage 2 — see struct docs for the destructive-op,
    /// hard-pin, and single-device constraints.
    pub fn with_fallback_placements_for(
        mut self,
        f: &'env (dyn Fn(DeviceLocation) -> Vec<(BackendId, DeviceLocation)> + 'env),
    ) -> Self {
        self.fallback_placements_for = Some(f);
        self
    }

    /// Attach a capabilities lookup. Required when `populate_costs`
    /// is true.
    pub fn with_capabilities_for(mut self, f: &'env CapabilitiesLookup<'env>) -> Self {
        self.capabilities_for = Some(f);
        self
    }

    /// Attach a Layer-2 Judge oracle. When set, `compile_plan`'s
    /// cost composer refines Layer-1 estimates with measured
    /// latencies per [`JudgeOracle::measured_latency_ns`].
    pub fn with_judge(mut self, judge: &'env dyn JudgeOracle) -> Self {
        self.judge = Some(judge);
        self
    }

    /// Attach a Stage-2 transfer-cost oracle. See the
    /// [`Self::transfer_estimator`] field docs — enables inbound-
    /// transfer pricing AND the priced off-device admission relax.
    /// Production wires `SystemTopology::estimate_transfer_ns`;
    /// tests use synthetic estimators.
    pub fn with_transfer_estimator(mut self, est: &'env dyn TransferEstimator) -> Self {
        self.transfer_estimator = Some(est);
        self
    }

    /// Attach a residency resolver for already-resident graph
    /// inputs (consts + persistent cache slots). See the
    /// [`Self::input_residency`] field docs.
    pub fn with_input_residency(
        mut self,
        f: &'env (dyn Fn(NodeId) -> Option<DeviceLocation> + 'env),
    ) -> Self {
        self.input_residency = Some(f);
        self
    }

    /// Attach a base plan for incremental extension (planner Stage
    /// 4). See the [`Self::reuse_plan`] field docs.
    pub fn with_reuse_plan(mut self, base: &'env ExecutionPlan) -> Self {
        self.reuse_plan = Some(base);
        self
    }
}

/// Build an [`ExecutionPlan`] from a topologically-ordered node
/// sequence + binding table + options.
///
/// For each node in `order`:
///
/// - If `op_to_op_kind(&node.op)` returns `None` (view ops,
///   `Op::Const`, ops not yet wired into dispatch), the node gets
///   no entry — view-only adoption + const adoption flow through
///   the legacy paths unchanged.
/// - `Op::Copy` / `Op::Move` nodes get no entry either (picker-arc
///   step 4a): their kernel backend is residency-determined (the
///   SOURCE backend), so transfer-kernel resolution stays with the
///   executor's `compile_node` path keyed by the bridge-maintained
///   source-backend stamp.
/// - Otherwise:
///   1. Resolve `(op_kind, dtypes, device)` from the graph + the
///      options (`device` via `Graph::placement`, then
///      `PlanOptions::pinned_device`, then
///      `default_device_for(target_backend)` for legacy stamped
///      graphs).
///   2. Build the placement list — either via
///      `options.placements_for_device` (cross-backend mode) or
///      `[(target_backend, device)]` (legacy single-backend mode).
///   3. Enumerate candidates via [`enumerate_candidates`].
///   4. Apply the default filter chain (precision floor +
///      strided-input pref + bit-stable pref).
///   5. If `populate_costs`, compute static costs via the
///      capabilities lookup — plus, when a
///      [`PlanOptions::transfer_estimator`] is configured, the
///      Stage-2 inbound-transfer term per candidate — and rank
///      ascending by composite cost.
///   6. Retain the per-ending-device Pareto frontier + crowding cap
///      ([`AlternativeSet::retain_per_device_frontier`], `keep =
///      KEEP_PER_DEVICE`).
///   7. If the set is empty after filtering, return
///      [`Error::NoBackendForOp`] (fail-fast at plan time, not
///      deep in executor dispatch).
///   8. Insert into `plan.alternatives`.
///
/// ## Stage-2 residency threading
///
/// `order` is topological, so each node's producers are planned —
/// and their placements committed — before the node itself ranks.
/// The walk maintains a per-node residency view mirroring the
/// bridge's `effective_placements` priority: residency-declaring
/// ops (`Op::Copy`/`Move`/`Alloc` targets) → explicit
/// `Graph::placement` → caller-supplied input residency
/// ([`PlanOptions::input_residency`], consts + persistent cache
/// slots) → the plan's own committed winner → backend stamp →
/// view-op pass-through inheritance. Inputs whose residency
/// resolves feed the inbound-transfer pricing; unknown residency
/// prices no term (conservative — an unpriceable edge neither
/// justifies nor penalizes a move).
///
/// ## Stage-3 carry-forward placement DP
///
/// Relaxed nodes whose admissible candidates span ≥ 2 devices defer
/// their winner choice to the carry-forward DP
/// ([`crate::ranker::placement_dp`]): the walk opens a state row
/// per such node (`best[d]` = accumulated cost arriving with output
/// on `d`), extends rows along the dominant chain (view-shaped
/// pass-throughs alias through), merges joins per-device by the
/// cheaper producer state, prices the exit crossing back to the
/// realize target on terminal rows, and backtracks the cheapest
/// terminal state into per-node device commitments. Each committed
/// row's `AlternativeSet` is then priced, ranked, and pruned to the
/// committed device — preserving the Stage-2 single-device-set
/// invariant the residency stitch depends on.
///
/// Gating: a row only opens when the node is relaxed (estimator +
/// fallback enumerator configured, no explicit placement, not
/// destructive, not already-resident) AND its candidates span more
/// than one device. Single-device plans never open a row and take
/// the Stage-2 greedy path bit-identically — zero overhead beyond
/// an empty-map check per node.
///
/// Documented debt (see the `placement_dp` module docs): fan-out
/// rows chain into their FIRST consumer only (later consumers price
/// that edge as unknown); diamonds re-anchor the second branch as a
/// fresh chain; hard-pinned consumers of an open row close it
/// toward their device at walk time, which prices the crossing but
/// cannot revisit it later.
pub fn compile_plan(
    graph: &Graph,
    order: &[NodeId],
    bindings_table: &KernelBindingTable,
    options: &PlanOptions<'_>,
) -> Result<ExecutionPlan> {
    // Snapshot the topology generation at plan-build time. The
    // executor (Phase 4.3) checks this against the live counter at
    // every dispatch-chunk boundary; mismatch surfaces
    // `Error::TopologyChanged` so the realize layer can rebuild
    // against the fresh topology.
    let generation = crate::dispatch::topology_generation();
    let mut alternatives_map = HashMap::with_capacity(order.len());

    // Stage-2 residency view, threaded along the walk (see the
    // doc comment). Nodes matching no resolution rule stay absent —
    // residency unknown, no transfer term fires on their edges.
    let mut residency: HashMap<NodeId, DeviceLocation> =
        HashMap::with_capacity(order.len());

    // Stage-3 DP state. Empty (and cost-free) unless multi-device
    // placement freedom actually materializes at some node.
    let mut dp = PlacementDp::new();
    let mut dp_drafts: HashMap<NodeId, NodeDraft> = HashMap::new();

    for &id in order {
        let node = graph.node(id);

        // Planner Stage 4: incremental-extension fast path. A node
        // already covered by the base plan reuses its stored set
        // verbatim; its residency commits with the same priority
        // order the full walk uses (early rules, then the stored
        // winner's device — the priority-4 "plan winner" rule).
        if let Some(base) = options.reuse_plan {
            if let Some(set) = base.alternatives(id) {
                let resident = early_residency(graph, options, id, node)
                    .or_else(|| set.winner().map(|w| w.device));
                if let Some(loc) = resident {
                    residency.insert(id, loc);
                }
                alternatives_map.insert(id, set.clone());
                continue;
            }
        }

        // Residency priorities 1–3: definitional transfer/alloc
        // targets, explicit placements, already-resident inputs.
        let early = early_residency(graph, options, id, node);

        let mut dp_row_opened = false;
        let winner_device = match build_node_draft(
            graph,
            id,
            node,
            bindings_table,
            options,
        )? {
            Some(draft) => {
                let devices = distinct_devices(&draft.set);
                let dp_estimator = options.transfer_estimator.filter(|_| {
                    draft.relaxed && early.is_none() && devices.len() >= 2
                });
                if let Some(est) = dp_estimator {
                    // Stage 3: open a DP row instead of committing a
                    // winner. Residency stays unresolved until the
                    // backtrack commits a device.
                    open_dp_row(&mut dp, graph, id, node, &draft.set, &devices, &residency, est);
                    dp_drafts.insert(id, draft);
                    dp_row_opened = true;
                    None
                } else {
                    // Stage-2 greedy finalize. When the node's device
                    // is already determined (single-device candidate
                    // set — hard pins, destructive ops, frozen
                    // fallbacks), close any open producer rows toward
                    // it FIRST so the close prices the crossing and
                    // the finalize prices the edge.
                    if let Some(est) = options.transfer_estimator {
                        if let [single] = devices.as_slice() {
                            close_open_inputs(
                                &mut dp, graph, node, *single, est, &mut residency,
                            );
                        }
                    }
                    let set = finalize_node_greedy(
                        graph,
                        node,
                        draft,
                        bindings_table,
                        options,
                        &residency,
                    )?;
                    let dev = set.winner().map(|w| w.device);
                    // Multi-device greedy sets (legacy missing-impl
                    // fallback shapes) learn their device only after
                    // ranking — close producers toward the winner.
                    if devices.len() > 1 {
                        if let (Some(est), Some(wd)) =
                            (options.transfer_estimator, dev)
                        {
                            close_open_inputs(
                                &mut dp, graph, node, wd, est, &mut residency,
                            );
                        }
                    }
                    alternatives_map.insert(id, set);
                    dev
                }
            }
            None => {
                // Structural / residency-determined nodes. An
                // `Op::Copy`/`Op::Move` consuming an open row anchors
                // the chain at its transfer target (the production
                // realize-root splice is exactly this shape — closing
                // here IS the exit pricing through an explicit node).
                if let fuel_graph::Op::Copy { target } | fuel_graph::Op::Move { target } =
                    node.op
                {
                    if let Some(est) = options.transfer_estimator {
                        close_open_inputs(
                            &mut dp, graph, node, target, est, &mut residency,
                        );
                    }
                }
                None
            }
        };

        if dp_row_opened {
            // Residency intentionally NOT committed — the DP owns
            // this node's placement; priorities 5–6 must not leak a
            // stamp-derived device downstream.
            continue;
        }

        // Commit this node's residency for downstream consumers.
        let resident = early
            .or(winner_device)
            .or_else(|| {
                // Priority 5: a backend stamp without a plan entry
                // (structural ops + legacy stamps) follows the
                // realize device, or the stamp's default device
                // when no pin is set.
                graph.target_backend(id).map(|b| {
                    options
                        .pinned_device
                        .unwrap_or_else(|| default_device_for(b))
                })
            })
            .or_else(|| {
                // Priority 6: residency-inheriting pass-throughs —
                // view ops, Reshape, Contiguize produce no new
                // residency; they follow their data input (already
                // resolved — the walk is topological).
                if is_passthrough(node) {
                    node.inputs.first().and_then(|i| residency.get(i)).copied()
                } else {
                    None
                }
            });
        if let Some(loc) = resident {
            residency.insert(id, loc);
        } else if is_passthrough(node) {
            // Stage 3: a pass-through of a DP-tracked node keeps the
            // chain connected — alias it to the producer's row so
            // consumers extend through the view. (If the row already
            // committed, surface its device as this view's residency,
            // mirroring priority 6.)
            if let Some(&input) = node.inputs.first() {
                let root = dp.resolve(input);
                if let Some(dev) = dp.committed_device(root) {
                    residency.insert(id, dev);
                } else {
                    dp.add_alias(id, root);
                }
            }
        }
    }

    // Stage 3: commit the DP (exit pricing + backtrack), then
    // finalize each deferred row against the committed placements.
    if !dp_drafts.is_empty() {
        let Some(est) = options.transfer_estimator else {
            // Rows only open with an estimator; this is unreachable
            // in practice but degrades typed rather than panicking.
            return Err(Error::Msg(
                "compile_plan: placement DP rows exist without a transfer \
                 estimator — internal invariant violated"
                    .into(),
            )
            .bt());
        };
        let commits =
            dp.finish(options.pinned_device, |nid| node_bytes(graph, nid), est);
        for (n, d) in commits {
            residency.insert(n, d);
        }
        // Pass-through residency for view chains the walk left
        // unresolved (views of DP nodes) — topo order, so producers
        // resolve before their views.
        for &id in order {
            if residency.contains_key(&id) {
                continue;
            }
            let node = graph.node(id);
            if is_passthrough(node) {
                if let Some(&loc) =
                    node.inputs.first().and_then(|i| residency.get(i))
                {
                    residency.insert(id, loc);
                }
            }
        }
        // Finalize deferred rows in topo order: price inbound
        // transfers from the final residencies, rank, prune to the
        // DP-committed device (Stage-2 single-device-set invariant),
        // truncate.
        for &id in order {
            let Some(draft) = dp_drafts.remove(&id) else {
                continue;
            };
            let chosen = dp.committed_device(id).ok_or_else(|| {
                Error::Msg(format!(
                    "compile_plan: placement DP left node {:?} uncommitted — \
                     internal invariant violated",
                    id,
                ))
                .bt()
            })?;
            let node = graph.node(id);
            let mut set = draft.set;
            let inputs = priced_inputs_for(graph, node, &residency);
            apply_inbound_transfer_costs(&mut set, &inputs, est);
            set.rank_by_composite_cost();
            let keep: Vec<usize> = set
                .alternatives()
                .iter()
                .enumerate()
                .filter_map(|(i, c)| (c.device == chosen).then_some(i))
                .collect();
            if keep.len() < set.len() {
                set.retain_indices(&keep);
            }
            // PR-B2: retire the fixed top-N. The set is now pruned to
            // the DP-committed single device, so the per-ending-device
            // Pareto frontier keeps that device's non-dominated paths
            // (crowding-capped at KEEP_PER_DEVICE); the arm-0 winner —
            // the realize pick — is preserved at index 0.
            set.retain_per_device_frontier(KEEP_PER_DEVICE);
            if set.is_empty() {
                // Defensive — the committed device came from the
                // row's own candidates, so this shouldn't fire.
                return Err(missing_binding_error(
                    bindings_table,
                    draft.op_kind,
                    &draft.dtypes,
                    draft.diag_backend,
                ));
            }
            alternatives_map.insert(id, set);
        }
    }

    Ok(ExecutionPlan {
        order: order.to_vec(),
        alternatives: alternatives_map,
        generation,
    })
}

/// Does this op receive an `AlternativeSet` entry from
/// [`compile_plan`]? The single source of truth for the plan-entry
/// gate (used by [`build_node_draft`] and `PlanOptions::reuse_plan`
/// incremental extension): `true` exactly when `op_to_op_kind` maps
/// the op AND it is not a residency-determined transfer (`Op::Copy` /
/// `Op::Move`, whose kernel backend the executor resolves from the
/// bridge-maintained source-backend stamp).
pub fn node_needs_plan_entry(op: &fuel_graph::Op) -> bool {
    op_to_op_kind(op).is_some()
        && !matches!(op, fuel_graph::Op::Copy { .. } | fuel_graph::Op::Move { .. })
}

/// Residency-inheriting pass-throughs: view ops, `Reshape`,
/// `Contiguize` produce no new residency — their bytes follow their
/// data input.
fn is_passthrough(node: &fuel_graph::Node) -> bool {
    node.op.is_view_op()
        || matches!(node.op, fuel_graph::Op::Reshape(_) | fuel_graph::Op::Contiguize)
}

/// Distinct candidate devices in first-seen order (the enumerator
/// lists the decision device first, so index 0 is the locality
/// anchor and DP argmin ties break toward it).
fn distinct_devices(set: &AlternativeSet) -> SmallVec<[DeviceLocation; 4]> {
    let mut out: SmallVec<[DeviceLocation; 4]> = SmallVec::new();
    for c in set.alternatives() {
        if !out.contains(&c.device) {
            out.push(c.device);
        }
    }
    out
}

/// Output bytes of a node — element count × dtype size, saturating.
/// Sub-byte dtypes price latency-only (same bounded undercount as
/// [`priced_inputs_for`]).
fn node_bytes(graph: &Graph, id: NodeId) -> u64 {
    let n = graph.node(id);
    (n.shape.elem_count() as u64).saturating_mul(n.dtype.size_in_bytes() as u64)
}

/// Stage 3: open a DP row for a relaxed multi-device node. Inputs
/// partition into known-fixed residencies (priced per device into
/// the row's base cost), open-row chain edges (extended via the DP
/// recurrence), and unknown edges (no term — conservative).
#[allow(clippy::too_many_arguments)]
fn open_dp_row(
    dp: &mut PlacementDp,
    graph: &Graph,
    id: NodeId,
    node: &fuel_graph::Node,
    set: &AlternativeSet,
    devices: &[DeviceLocation],
    residency: &HashMap<NodeId, DeviceLocation>,
    est: &dyn TransferEstimator,
) {
    // Per-device node cost: the cheapest same-device candidate's
    // composite (static + Judge-refined) figure.
    let device_costs: Vec<(DeviceLocation, u64)> = devices
        .iter()
        .map(|&d| {
            let min = set
                .alternatives()
                .iter()
                .filter(|c| c.device == d)
                .map(|c| composite_ns(&c.static_cost))
                .min()
                .unwrap_or(u64::MAX);
            (d, min)
        })
        .collect();

    let mut fixed: Vec<(DeviceLocation, u64)> = Vec::new();
    let mut chains: Vec<ChainInput> = Vec::new();
    for &input in &node.inputs {
        let bytes = node_bytes(graph, input);
        if let Some(&src) = residency.get(&input) {
            fixed.push((src, bytes));
            continue;
        }
        let root = dp.resolve(input);
        if dp.is_open(root) {
            match chains.iter_mut().find(|ci| ci.producer == root) {
                Some(ci) => ci.edge_bytes.push(bytes),
                None => chains.push(ChainInput {
                    producer: root,
                    edge_bytes: smallvec::smallvec![bytes],
                }),
            }
        } else if let Some(dev) = dp.committed_device(root) {
            // Producer row already closed toward an earlier consumer
            // — its device is fixed now.
            fixed.push((dev, bytes));
        }
        // else: chained-but-uncommitted or genuinely unknown — no
        // term (documented fan-out debt).
    }
    dp.push_row(id, &device_costs, &fixed, &chains, est);
}

/// Close any open producer rows feeding `node` toward `toward` —
/// the consumer's (already-known) device. Commitments merge into
/// the walk's residency view so the consumer's own pricing and all
/// later consumers see them; direct inputs that alias to a
/// committed row inherit its device (pass-through residency).
fn close_open_inputs(
    dp: &mut PlacementDp,
    graph: &Graph,
    node: &fuel_graph::Node,
    toward: DeviceLocation,
    est: &dyn TransferEstimator,
    residency: &mut HashMap<NodeId, DeviceLocation>,
) {
    for &input in &node.inputs {
        if residency.contains_key(&input) {
            continue;
        }
        let root = dp.resolve(input);
        if dp.is_open(root) {
            let commits =
                dp.close_toward(root, toward, &[node_bytes(graph, input)], est);
            for (n, d) in commits {
                residency.insert(n, d);
            }
        }
        if !residency.contains_key(&input) {
            if let Some(d) = dp.committed_device(root) {
                residency.insert(input, d);
            }
        }
    }
}

/// Stage-2 residency, priorities 1–3: rules that resolve a node's
/// placement WITHOUT the plan — residency-declaring ops
/// (`Op::Copy`/`Move`/`Alloc` carry their output's location in the
/// op variant), explicit `Graph::placement` (scheduler
/// assignments), and the caller's already-resident input lookup
/// (consts + persistent cache slots via
/// [`PlanOptions::input_residency`]). Mirrors the top of the
/// bridge's `effective_placements` priority list.
fn early_residency(
    graph: &Graph,
    options: &PlanOptions<'_>,
    id: NodeId,
    node: &fuel_graph::Node,
) -> Option<DeviceLocation> {
    match node.op {
        fuel_graph::Op::Copy { target }
        | fuel_graph::Op::Move { target }
        | fuel_graph::Op::Alloc { target } => return Some(target),
        _ => {}
    }
    if let Some(loc) = graph.placement(id) {
        return Some(loc);
    }
    options.input_residency.and_then(|f| f(id))
}

/// Collect `(resident device, byte size)` for every input of `node`
/// whose residency is known. Bytes = element count × dtype size.
/// Sub-byte dtypes (`size_in_bytes() == 0`) price latency-only — a
/// bounded undercount, acceptable until a packed-bit byte-size
/// helper exists (no such dtype crosses devices today).
fn priced_inputs_for(
    graph: &Graph,
    node: &fuel_graph::Node,
    residency: &HashMap<NodeId, DeviceLocation>,
) -> Vec<(DeviceLocation, u64)> {
    node.inputs
        .iter()
        .filter_map(|&input_id| {
            let src = residency.get(&input_id).copied()?;
            let in_node = graph.node(input_id);
            let bytes = (in_node.shape.elem_count() as u64)
                .saturating_mul(in_node.dtype.size_in_bytes() as u64);
            Some((src, bytes))
        })
        .collect()
}

/// The residency-independent half of one node's plan: the
/// enumerated + filtered + statically-costed candidate set, plus
/// the flags the finalize step (greedy or DP) needs. Produced by
/// [`build_node_draft`]; consumed by [`finalize_node_greedy`] or
/// deferred into the Stage-3 DP.
struct NodeDraft {
    set: AlternativeSet,
    /// Stage-2 priced-admission regime (estimator + fallback +
    /// non-destructive + no explicit placement).
    relaxed: bool,
    /// Set was produced by the legacy missing-impl fallback —
    /// freezes to its single ranked winner.
    from_fallback: bool,
    /// Static costs (Layer 1 + Judge Layer 2) were computed —
    /// gates transfer pricing + ranking in the finalize step,
    /// mirroring the `populate_costs && capabilities_for` gate.
    costed: bool,
    op_kind: OpKind,
    dtypes: Vec<DType>,
    diag_backend: BackendId,
}

/// `compile_plan`'s per-node enumeration phase: enumeration +
/// filter chain + static cost composition for one kernel-bearing
/// node — everything that does NOT depend on producer residency.
/// Returns `Ok(None)` for nodes that get no plan entry (view ops,
/// `Op::Const`, residency-determined `Op::Copy`/`Op::Move`, ops
/// without a dispatch OpKind).
fn build_node_draft(
    graph: &Graph,
    id: NodeId,
    node: &fuel_graph::Node,
    bindings_table: &KernelBindingTable,
    options: &PlanOptions<'_>,
) -> Result<Option<NodeDraft>> {
    // Gate shared with the plan store's coverage check
    // ([`node_needs_plan_entry`]): no OpKind mapping → no entry;
    // Op::Copy / Op::Move are residency-determined, not picker
    // decisions — their kernel runs on the backend that owns the
    // SOURCE bytes (`copy_from_cpu_wrapper` for H2D, the source
    // backend's download wrapper for D2H). Enumerating them
    // against a single decision device would key the lookup at
    // the wrong end of the transfer for placement-carrying
    // copies (the consumer device, where the H2D copy's kernel
    // does NOT run). The executor's legacy `compile_node` path
    // resolves these via the source-backend `target_backend`
    // stamp maintained by the bridge's copy-insertion passes.
    // Picker-arc step 4a; Op::Move added with the executor's
    // Move arm (Op::Move maps to OpKind::Copy — same kernel,
    // destructive release is realize-loop bookkeeping).
    if !node_needs_plan_entry(&node.op) {
        return Ok(None);
    }
    let Some(op_kind) = op_to_op_kind(&node.op) else {
        // Unreachable given the gate above; keep the typed shape.
        return Ok(None);
    };

    // Decision-device resolution (step 4a): explicit per-node
    // placement (scheduler assignments) → the realize call's
    // pinned device → the legacy stamped-backend default.
    let explicit_backend = graph.target_backend(id);
    let target_device = graph
        .placement(id)
        .or(options.pinned_device)
        .or_else(|| explicit_backend.map(default_device_for))
        .ok_or_else(|| {
            Error::Msg(format!(
                "compile_plan: node {:?} ({:?}) has no device context — \
                 set PlanOptions::pinned_device, a graph placement, or \
                 the node's target_backend",
                id, node.op,
            ))
            .bt()
        })?;
    // Diagnostic backend for error reporting; also the legacy
    // single-backend placement when no topology enumerator is
    // configured.
    let diag_backend =
        explicit_backend.unwrap_or_else(|| backend_for_device(target_device));
    let dtypes = build_lookup_dtypes(graph, node);

    // Build the placement list: cross-backend if a topology
    // enumerator is configured, otherwise single-backend
    // legacy mode.
    let placements: Vec<(BackendId, DeviceLocation)> =
        match options.placements_for_device {
            Some(f) => f(target_device)
                .into_iter()
                .map(|backend| (backend, target_device))
                .collect(),
            None => vec![(diag_backend, target_device)],
        };

    let op_params = candidate_default_op_params(graph, node);

    // Stage-2 admission relax: when inbound-transfer pricing is
    // active, off-device candidates ALWAYS enumerate — priced;
    // locality emerges from the numbers, not from the legacy
    // missing-impl-only gate. Three hard exclusions keep the
    // relax sound:
    //
    // - **Explicit `Graph::placement` is a hard pin.** A per-node
    //   placement is a scheduler / profiling decision (the Judge
    //   measuring a specific backend pins its profile graphs this
    //   way) — the planner must not silently move it to a
    //   "cheaper" sibling device. The realize-call `pinned_device`
    //   is a soft default the planner may improve on.
    // - **Destructive ops never move** off the device that owns
    //   their mutation target.
    // - **Pricing must actually be active** (estimator +
    //   populate_costs + capabilities): an unpriced rank would
    //   move ops on kernel cost alone, regressing locality.
    let pricing_active = options.transfer_estimator.is_some()
        && options.populate_costs
        && options.capabilities_for.is_some();
    let fallback_allowed = node.op.destructive_input().is_none()
        && options.fallback_placements_for.is_some();
    let relaxed =
        pricing_active && fallback_allowed && graph.placement(id).is_none();

    let mut set;
    let mut from_fallback = false;
    if relaxed {
        // Merged enumeration: decision-device placements FIRST so
        // the rank's stable sort breaks zero-signal ties toward
        // locality, then every off-device placement — all priced
        // by the same cost + transfer composition.
        let mut merged = placements.clone();
        if let Some(fallback) = options.fallback_placements_for {
            merged.extend(fallback(target_device));
        }
        set = enumerate_candidates(
            op_kind,
            &dtypes,
            &merged,
            &op_params,
            bindings_table,
        );
    } else {
        set = enumerate_candidates(
            op_kind,
            &dtypes,
            &placements,
            &op_params,
            bindings_table,
        );

        // Picker-arc step 4b (legacy, unpriced path): when the
        // decision device has NO implementation for this
        // (op, dtypes), admit off-device candidates from the
        // fallback enumerator. The missing-impl op becomes a
        // plan-time picker decision (the bridge stitches residency
        // via Op::Copy insertion around the off-device winner)
        // instead of a realize-time error.
        if set.is_empty() && fallback_allowed {
            if let Some(fallback) = options.fallback_placements_for {
                let fb_placements = fallback(target_device);
                if !fb_placements.is_empty() {
                    set = enumerate_candidates(
                        op_kind,
                        &dtypes,
                        &fb_placements,
                        &op_params,
                        bindings_table,
                    );
                    from_fallback = !set.is_empty();
                }
            }
        }
    }

    // Fail-fast: if enumeration found nothing (on the decision
    // device AND via fallback), surface the missing-binding
    // error before filters can also empty the set (which would
    // produce a less-specific FilterRejected).
    if set.is_empty() {
        return Err(missing_binding_error(
            bindings_table,
            op_kind,
            &dtypes,
            diag_backend,
        ));
    }

    // Apply the default filter chain.
    let input_layouts: Vec<fuel_ir::Layout> = node
        .inputs
        .iter()
        .map(|&input_id| graph.layout(input_id))
        .collect();
    let ctx = FilterContext::new(op_kind, &dtypes, &input_layouts);
    let chain = default_chain(options.precision_requirement);
    if let Err(err) = apply_filter_chain(&mut set, &chain, &ctx) {
        // Stage-2 fix of the picker-4b verifier minor: a decision
        // device whose registrations all FAIL the hard filter
        // chain falls back exactly like an empty enumeration —
        // the pin can't deliver an admissible kernel either way.
        // Only meaningful when the rejected set couldn't already
        // contain off-device candidates: relaxed merged sets and
        // fallback sets rejected here are genuine global
        // rejections and propagate.
        if relaxed || from_fallback || !fallback_allowed {
            return Err(err);
        }
        let Some(fallback) = options.fallback_placements_for else {
            return Err(err);
        };
        let fb_placements = fallback(target_device);
        if fb_placements.is_empty() {
            return Err(err);
        }
        let mut fb_set = enumerate_candidates(
            op_kind,
            &dtypes,
            &fb_placements,
            &op_params,
            bindings_table,
        );
        if fb_set.is_empty() {
            return Err(err);
        }
        // The fallback set faces the same chain — a hard reject
        // here is the surfaceable error (nowhere admissible).
        apply_filter_chain(&mut fb_set, &chain, &ctx)?;
        set = fb_set;
        from_fallback = true;
    }

    // Stamp the decision-point identity so dispatch-time
    // selectors (Picker 2) can re-query the Judge per candidate.
    // The derivation mirrors `compute_static_costs`'s Layer-2
    // lookup key exactly: principal dtype = first lookup dtype,
    // size class = first input's element count (SizeClass(0)
    // for nullary ops). Runs after the filter/fallback resolution
    // so a filter-rejection fallback set is stamped too.
    if let Some(&principal_dtype) = dtypes.first() {
        let size_class = node
            .inputs
            .first()
            .map(|&input_id| {
                SizeClass::from_elem_count(graph.node(input_id).shape.elem_count())
            })
            .unwrap_or(SizeClass(0));
        set.set_context(DecisionContext {
            op: op_kind,
            principal_dtype,
            size_class,
        });
    }

    // Static cost composition (optional — tests may skip). Layer-1
    // via the binding-table CostFns + Layer-2 Judge refinement;
    // both residency-independent, so they belong to the draft.
    let costed = if options.populate_costs {
        if let Some(caps_for) = options.capabilities_for {
            let input_shapes: Vec<Shape> = node
                .inputs
                .iter()
                .map(|&input_id| graph.node(input_id).shape.clone())
                .collect();
            compute_static_costs(
                &mut set,
                op_kind,
                &dtypes,
                &input_shapes,
                bindings_table,
                caps_for,
                options.judge,
            );
            true
        } else {
            false
        }
    } else {
        false
    };

    Ok(Some(NodeDraft {
        set,
        relaxed,
        from_fallback,
        costed,
        op_kind,
        dtypes,
        diag_backend,
    }))
}

/// `compile_plan`'s per-node commit phase, Stage-2 greedy regime:
/// inbound-transfer pricing against the committed producer
/// residencies, rank, relaxed winner-device prune, truncation, and
/// the fallback freeze. The Stage-3 DP path replaces this for
/// deferred rows (pricing + prune happen after the backtrack
/// instead).
fn finalize_node_greedy(
    graph: &Graph,
    node: &fuel_graph::Node,
    draft: NodeDraft,
    bindings_table: &KernelBindingTable,
    options: &PlanOptions<'_>,
    residency: &HashMap<NodeId, DeviceLocation>,
) -> Result<AlternativeSet> {
    let NodeDraft {
        mut set,
        relaxed,
        from_fallback,
        costed,
        op_kind,
        dtypes,
        diag_backend,
    } = draft;

    if costed {
        // Stage 2: price the transfers each candidate's placement
        // would induce from its inputs' committed residencies. Runs
        // AFTER the Layer-2 Judge refinement (which REPLACES the
        // kernel-time estimate) so the transfer term survives; the
        // rank adds the two.
        if let Some(estimator) = options.transfer_estimator {
            let inputs = priced_inputs_for(graph, node, residency);
            apply_inbound_transfer_costs(&mut set, &inputs, estimator);
        }
        set.rank_by_composite_cost();
    }

    // Stage 2: after the priced rank, prune the relaxed set to the
    // winner's device. The bridge's residency stitch (Op::Copy
    // insertion) is a graph rewrite computed from the static
    // winner; cross-device siblings left in the set would let a
    // dispatch-time selector pick a candidate whose inputs were
    // never copied to its device. Same-device siblings stay — the
    // stitch covers all of them, and the transfer term is uniform
    // across one device so the selector's relative ranking is
    // unaffected. Pruned BEFORE truncation so same-device
    // alternatives fill the top-N.
    if relaxed {
        if let Some(winner_device) = set.winner().map(|c| c.device) {
            let keep: Vec<usize> = set
                .alternatives()
                .iter()
                .enumerate()
                .filter_map(|(i, c)| (c.device == winner_device).then_some(i))
                .collect();
            if keep.len() < set.len() {
                set.retain_indices(&keep);
            }
        }
    }

    // PR-B2: retire the fixed top-N. Retain the per-ending-device
    // Pareto frontier (crowding-capped at KEEP_PER_DEVICE) instead of
    // a global top-N. The arm-0 winner — the realize pick — stays at
    // index 0. In the `relaxed` path the set was just pruned to the
    // winner's single device, so this reduces to that device's Pareto
    // front; in the legacy multi-backend-at-one-device path it keeps
    // the non-dominated kernels at the decision device.
    set.retain_per_device_frontier(KEEP_PER_DEVICE);

    // Step 4b: off-device fallback sets freeze to their single
    // ranked winner. The bridge's residency stitch (Op::Copy
    // insertion) is a graph rewrite computed from the static
    // winner; leaving siblings on OTHER devices in the set
    // would let a dispatch-time selector pick a candidate whose
    // inputs were never copied to its device.
    if from_fallback && set.len() > 1 {
        set.retain_indices(&[0]);
    }

    // After retention an empty set is the surfaceable error
    // (the filter chain or rank dropped everything). The chain
    // would have raised FilterRejected on hard-empty, so this
    // path is only reachable when populate_costs + ranking
    // pruned a structurally-empty set, which shouldn't happen
    // in practice — defensive only.
    if set.is_empty() {
        return Err(missing_binding_error(
            bindings_table,
            op_kind,
            &dtypes,
            diag_backend,
        ));
    }

    Ok(set)
}

/// Build an `Error::NoBackendForOp` diagnostic for a decision
/// point with no registered alternative. Same shape as
/// [`crate::kernel::KernelBindingTable::lookup_with_caps`]'s error
/// branch so callers see consistent output regardless of which
/// path surfaced the miss.
fn missing_binding_error(
    table: &KernelBindingTable,
    op: OpKind,
    dtypes: &[DType],
    backend: BackendId,
) -> Error {
    let _ = (table, backend);
    Error::NoBackendForOp {
        op,
        dtypes: dtypes.to_vec(),
        available_backends: Vec::new(),
        supported_combinations: Vec::new(),
    }
    .bt()
}

/// Helper that produces the `OpParams` to attach to enumerated
/// candidates for one graph node. Phase 1.5 ships with the
/// placeholder `OpParams::None` for every node — the live
/// op-params shape derivation lives in `pipelined::op_to_op_params`
/// but is currently mid-refactor for other reasons. Phase 4's
/// executor integration replaces this with the live derivation so
/// the planner's candidate set matches what dispatch sees.
fn candidate_default_op_params(
    _graph: &Graph,
    _node: &fuel_graph::Node,
) -> crate::kernel::OpParams {
    crate::kernel::OpParams::None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::PrecisionGuarantee;
    use crate::kernel::{unknown_cost, KernelCaps, OpParams};
    use fuel_ir::backend::{
        BackendCapabilities, SubstrateClass, TransferPath,
    };
    use fuel_ir::{DType, Layout, Result as FuelResult, Shape, StrideVec};
    use fuel_graph::{topo_order, Node, Op};
    use fuel_memory::Storage;
    use smallvec::smallvec;
    use std::collections::HashSet;
    use std::sync::{Arc, RwLock};

    fn noop_kernel(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> FuelResult<()> {
        Ok(())
    }

    fn noop_kernel_b(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> FuelResult<()> {
        Ok(())
    }

    fn noop_kernel_c(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> FuelResult<()> {
        Ok(())
    }

    fn register_add_f32(
        table: &mut KernelBindingTable,
        backend: BackendId,
        kernel: crate::kernel::KernelRef,
        precision: PrecisionGuarantee,
    ) {
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            backend,
            kernel,
            KernelCaps::empty(),
            precision,
            unknown_cost,
        );
    }

    fn build_add_graph() -> (Graph, NodeId) {
        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let rhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let add = g.push(Node {
            op: Op::Add,
            inputs: vec![lhs, rhs],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        g.set_target_backend(add, BackendId::Cpu);
        (g, add)
    }

    fn cpu_caps() -> BackendCapabilities {
        BackendCapabilities {
            backend_id: BackendId::Cpu,
            device_location: DeviceLocation::Cpu,
            op_dtype_support: HashSet::new(),
            required_alignment: 64,
            access_granularity_bits: 8,
            transfer_paths: vec![(DeviceLocation::Cpu, TransferPath::SameDevice)],
            storage_substrate: SubstrateClass::HostBytes,
        }
    }

    #[test]
    fn empty_plan_has_no_alternatives() {
        let p = ExecutionPlan::empty();
        assert!(p.order.is_empty());
        assert!(p.alternatives.is_empty());
        assert!(p.alternatives(NodeId(0)).is_none());
    }

    #[test]
    fn default_device_per_backend_matches_router_convention() {
        assert_eq!(default_device_for(BackendId::Cpu), DeviceLocation::Cpu);
        assert_eq!(
            default_device_for(BackendId::Cuda),
            DeviceLocation::Cuda { gpu_id: 0 },
        );
        assert_eq!(
            default_device_for(BackendId::Vulkan),
            DeviceLocation::Vulkan { gpu_id: 0 },
        );
    }

    #[test]
    fn compile_plan_skips_view_and_const_nodes() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );

        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[6]),
            dtype: DType::F32,
        });
        let rhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[6]),
            dtype: DType::F32,
        });
        let add = g.push(Node {
            op: Op::Add,
            inputs: vec![lhs, rhs],
            shape: Shape::from_dims(&[6]),
            dtype: DType::F32,
        });
        let reshape = g.push(Node {
            op: Op::Reshape(Shape::from_dims(&[2, 3])),
            inputs: vec![add],
            shape: Shape::from_dims(&[2, 3]),
            dtype: DType::F32,
        });
        g.set_target_backend(add, BackendId::Cpu);
        g.set_target_backend(reshape, BackendId::Cpu);

        let order = topo_order(&g, reshape);
        let opts = PlanOptions::new().without_cost_population();
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");

        assert_eq!(plan.order.len(), 4);
        assert_eq!(
            plan.alternatives.len(),
            1,
            "only Add carries an alternative set",
        );
        let alts = plan.alternatives(add).expect("Add alternatives present");
        assert_eq!(alts.len(), 1);
        assert!(plan.alternatives(lhs).is_none());
        assert!(plan.alternatives(rhs).is_none());
        assert!(plan.alternatives(reshape).is_none());
    }

    #[test]
    fn compile_plan_fails_fast_on_missing_binding() {
        let table = KernelBindingTable::new();
        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[2, 3]),
            dtype: DType::F32,
        });
        let rhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3, 2]),
            dtype: DType::F32,
        });
        let mm = g.push(Node {
            op: Op::MatMul,
            inputs: vec![lhs, rhs],
            shape: Shape::from_dims(&[2, 2]),
            dtype: DType::F32,
        });
        g.set_target_backend(mm, BackendId::Cpu);

        let order = topo_order(&g, mm);
        let opts = PlanOptions::new().without_cost_population();
        let err = compile_plan(&g, &order, &table, &opts).unwrap_err();
        match err {
            Error::NoBackendForOp { op, dtypes, .. } => {
                assert_eq!(op, OpKind::MatMul);
                assert_eq!(dtypes, vec![DType::F32, DType::F32, DType::F32]);
            }
            other => panic!("expected NoBackendForOp, got {other:?}"),
        }
    }

    #[test]
    fn compile_plan_legacy_single_backend_uses_target_backend() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        // Even with an AOCL alternative registered, the legacy mode
        // (placements_for_device = None) only uses the node's
        // target_backend (Cpu).
        register_add_f32(
            &mut table,
            BackendId::Cuda,
            noop_kernel_b,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let opts = PlanOptions::new().without_cost_population();
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add_id).unwrap();
        assert_eq!(alts.len(), 1, "legacy mode → single backend → single alt");
        assert_eq!(alts.winner().unwrap().backend, BackendId::Cpu);
    }

    #[test]
    fn compile_plan_cross_co_located_via_placements_callback() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        register_add_f32(
            &mut table,
            BackendId::Cuda,
            noop_kernel_b,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );

        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu {
                vec![BackendId::Cpu, BackendId::Cuda]
            } else {
                vec![]
            }
        };
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_placements_for_device(&placements_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add_id).unwrap();
        assert_eq!(
            alts.len(),
            2,
            "cross-co-located mode aggregates Cpu + Aocl alternatives",
        );
    }

    #[test]
    fn compile_plan_precision_requirement_filters_via_default_chain() {
        let mut table = KernelBindingTable::new();
        // Non-bit-stable.
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee {
                bit_stable_on_same_hardware: false,
                ..PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU
            },
        );

        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_precision_requirement(PrecisionRequirement::BIT_STABLE);
        let err = compile_plan(&g, &order, &table, &opts).unwrap_err();
        match err {
            Error::FilterRejected { filter, .. } => {
                assert_eq!(filter, "precision-floor");
            }
            other => panic!("expected FilterRejected, got {other:?}"),
        }
        // Sanity: drop the requirement → plan succeeds.
        let unconstrained = PlanOptions::new().without_cost_population();
        let plan = compile_plan(&g, &order, &table, &unconstrained).expect("compile");
        assert!(plan.alternatives(add_id).is_some());
        let _ = add_id;
    }

    #[test]
    fn compile_plan_with_cost_ranks_cheaper_winner() {
        fn cheap_cost(
            _: &[Shape],
            _: &[DType],
            _: &OpParams,
            _: &BackendCapabilities,
        ) -> crate::fused::CostEstimate {
            crate::fused::CostEstimate { flops: 10, bytes_moved: 0, kernel_overhead_ns: 0 }
        }
        fn expensive_cost(
            _: &[Shape],
            _: &[DType],
            _: &OpParams,
            _: &BackendCapabilities,
        ) -> crate::fused::CostEstimate {
            crate::fused::CostEstimate {
                flops: 1_000_000,
                bytes_moved: 0,
                kernel_overhead_ns: 0,
            }
        }
        let mut table = KernelBindingTable::new();
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            noop_kernel,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            expensive_cost,
        );
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cuda,
            noop_kernel_b,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            cheap_cost,
        );

        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu {
                vec![BackendId::Cpu, BackendId::Cuda]
            } else {
                vec![]
            }
        };
        let cpu_caps_val = cpu_caps();
        let caps_fn = |_: BackendId| Some(&cpu_caps_val);
        let opts = PlanOptions::new()
            .with_placements_for_device(&placements_fn)
            .with_capabilities_for(&caps_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add_id).unwrap();
        assert_eq!(alts.len(), 2);
        assert_eq!(
            alts.winner().unwrap().backend,
            BackendId::Cuda,
            "cheaper static cost wins after ranking",
        );
    }

    #[test]
    fn compile_plan_judge_layer2_can_flip_layer1_winner() {
        // Layer-1 says CPU is cheap, Aocl is expensive.
        // Layer-2 (Judge) measured the opposite: Aocl 20 ns, CPU 5000 ns.
        // After compile_plan's cost composition + rank, Aocl wins.
        use crate::ranker::HashMapJudge;
        use fuel_ir::dispatch::SizeClass;

        fn cpu_layer1(
            _: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities,
        ) -> crate::fused::CostEstimate {
            crate::fused::CostEstimate { flops: 10, bytes_moved: 0, kernel_overhead_ns: 0 }
        }
        fn aocl_layer1(
            _: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities,
        ) -> crate::fused::CostEstimate {
            crate::fused::CostEstimate { flops: 100_000, bytes_moved: 0, kernel_overhead_ns: 0 }
        }
        let mut table = KernelBindingTable::new();
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu, noop_kernel, KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU, cpu_layer1,
        );
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cuda, noop_kernel_b, KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU, aocl_layer1,
        );

        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu {
                vec![BackendId::Cpu, BackendId::Cuda]
            } else { vec![] }
        };
        let cpu_caps_val = cpu_caps();
        let caps_fn = |_: BackendId| Some(&cpu_caps_val);

        // Each input is shape [3] (per build_add_graph); the
        // first-input-shape rule gives elem_count 3.
        let sc = SizeClass::from_elem_count(3);
        let mut judge = HashMapJudge::new();
        judge.insert(OpKind::AddElementwise, DType::F32, sc, BackendId::Cpu, "", 5_000);
        judge.insert(OpKind::AddElementwise, DType::F32, sc, BackendId::Cuda, "", 20);

        let opts = PlanOptions::new()
            .with_placements_for_device(&placements_fn)
            .with_capabilities_for(&caps_fn)
            .with_judge(&judge);

        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add_id).unwrap();
        assert_eq!(alts.len(), 2);
        assert_eq!(
            alts.winner().unwrap().backend,
            BackendId::Cuda,
            "Layer-2 measurement reverses Layer-1 verdict via compile_plan",
        );
    }

    /// `compile_plan` stamps each set with the decision-point
    /// context using the same key derivation as the cost composer's
    /// Layer-2 lookup: principal dtype = first lookup dtype, size
    /// class = first input's element count.
    #[test]
    fn compile_plan_stamps_decision_context() {
        use fuel_ir::dispatch::SizeClass;

        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let opts = PlanOptions::new().without_cost_population();
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add_id).expect("Add alternatives present");
        let ctx = alts.context().expect("context stamped");
        assert_eq!(ctx.op, OpKind::AddElementwise);
        assert_eq!(ctx.principal_dtype, DType::F32);
        // build_add_graph inputs are shape [3].
        assert_eq!(ctx.size_class, SizeClass::from_elem_count(3));
    }

    /// PR-B2: the fixed top-N is retired. Three distinct backends
    /// co-located at one ending device, all equally-costed (no cost
    /// population → equal cost vectors → mutually non-dominated), all
    /// survive the per-ending-device Pareto frontier — the
    /// never-strand-last-`(device, backend)` rule keeps each distinct
    /// backend, and the crowding cap (KEEP_PER_DEVICE = 32) is far
    /// above 3. The old `truncate_to_top_n` would have stranded one;
    /// this is the behavior change B2 introduces.
    #[test]
    fn compile_plan_retains_per_device_frontier_no_top_n() {
        let mut table = KernelBindingTable::new();
        // Three CPU-substrate backends competing at one decision point.
        // Distinct kernel fn items so the binding table accepts them.
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        register_add_f32(
            &mut table,
            BackendId::Cuda,
            noop_kernel_b,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        register_add_f32(
            &mut table,
            BackendId::Vulkan,
            noop_kernel_c,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );

        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu {
                vec![
                    BackendId::Cpu,
                    BackendId::Cuda,
                    BackendId::Vulkan,
                ]
            } else {
                vec![]
            }
        };
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_placements_for_device(&placements_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add_id).unwrap();
        assert_eq!(
            alts.len(),
            3,
            "all three co-located (device, backend) paths survive — no top-N truncation",
        );
    }

    /// Step 4a parity: an UNSTAMPED graph planned with
    /// `with_pinned_device(Cpu)` produces the same alternative sets
    /// as the legacy stamped graph — candidate-for-candidate
    /// (backend, device, kernel, kernel_source).
    #[test]
    fn compile_plan_pinned_device_matches_stamped_plan() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );

        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu {
                vec![BackendId::Cpu]
            } else {
                vec![]
            }
        };

        // Legacy: stamped target_backend, no pinned device.
        let (g_stamped, add_stamped) = build_add_graph();
        let order_stamped = topo_order(&g_stamped, add_stamped);
        let opts_stamped = PlanOptions::new()
            .without_cost_population()
            .with_placements_for_device(&placements_fn);
        let plan_stamped =
            compile_plan(&g_stamped, &order_stamped, &table, &opts_stamped)
                .expect("stamped compile");

        // New: same graph shape, NO target_backend anywhere, pinned
        // device instead.
        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let rhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let add = g.push(Node {
            op: Op::Add,
            inputs: vec![lhs, rhs],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let order = topo_order(&g, add);
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_placements_for_device(&placements_fn)
            .with_pinned_device(DeviceLocation::Cpu);
        let plan = compile_plan(&g, &order, &table, &opts).expect("pinned compile");

        let a = plan_stamped.alternatives(add_stamped).expect("stamped set");
        let b = plan.alternatives(add).expect("pinned set");
        assert_eq!(a.len(), b.len(), "same candidate count");
        for (ca, cb) in a.alternatives().iter().zip(b.alternatives()) {
            assert_eq!(ca.backend, cb.backend);
            assert_eq!(ca.device, cb.device);
            assert_eq!(ca.kernel as usize, cb.kernel as usize, "same kernel ref");
            assert_eq!(ca.kernel_source, cb.kernel_source);
        }
        assert_eq!(a.context(), b.context(), "same decision context");
    }

    /// Step 4a: a node with neither a placement, nor a pinned
    /// device, nor a target_backend stamp is a plan-time error —
    /// fail-fast with an actionable message.
    #[test]
    fn compile_plan_errors_without_any_device_context() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let add = g.push(Node {
            op: Op::Add,
            inputs: vec![lhs, lhs],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let order = topo_order(&g, add);
        let opts = PlanOptions::new().without_cost_population();
        let err = compile_plan(&g, &order, &table, &opts).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("no device context"),
            "expected the device-context diagnostic, got: {msg}",
        );
    }

    /// Step 4a: `Op::Copy` nodes are excluded from the plan map —
    /// their kernel is residency-determined (source backend), so
    /// dispatch flows through the executor's legacy `compile_node`
    /// path keyed by the bridge's source-backend stamp.
    #[test]
    fn compile_plan_skips_op_copy_nodes() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        // A Copy registration exists — the skip must not depend on
        // the table lacking one.
        table.register_full(
            OpKind::Copy,
            &[DType::F32, DType::F32],
            BackendId::Cpu,
            noop_kernel_b,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            unknown_cost,
        );

        let (mut g, add_id) = build_add_graph();
        let copy_id = g.push(Node {
            op: Op::Copy { target: DeviceLocation::Cpu },
            inputs: vec![add_id],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        g.set_target_backend(copy_id, BackendId::Cpu);
        let order = topo_order(&g, copy_id);
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_pinned_device(DeviceLocation::Cpu);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        assert!(plan.alternatives(add_id).is_some(), "Add is planned");
        assert!(
            plan.alternatives(copy_id).is_none(),
            "Op::Copy is residency-determined — no plan entry",
        );
    }

    /// `Op::Move` follows `Op::Copy`'s exclusion: it dispatches the
    /// same residency-determined transfer kernel (`OpKind::Copy` on
    /// the SOURCE backend), so it gets no plan entry either.
    #[test]
    fn compile_plan_skips_op_move_nodes() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        // A Copy registration exists (Op::Move keys the same row) —
        // the skip must not depend on the table lacking one.
        table.register_full(
            OpKind::Copy,
            &[DType::F32, DType::F32],
            BackendId::Cpu,
            noop_kernel_b,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            unknown_cost,
        );

        let (mut g, add_id) = build_add_graph();
        let move_id = g.push(Node {
            op: Op::Move { target: DeviceLocation::Cpu },
            inputs: vec![add_id],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        g.set_target_backend(move_id, BackendId::Cpu);
        let order = topo_order(&g, move_id);
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_pinned_device(DeviceLocation::Cpu);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        assert!(plan.alternatives(add_id).is_some(), "Add is planned");
        assert!(
            plan.alternatives(move_id).is_none(),
            "Op::Move is residency-determined — no plan entry",
        );
    }

    /// Step 4b: an op with no implementation on the pinned device
    /// falls back to an off-device candidate — the missing-impl op
    /// becomes a plan-time picker decision.
    #[test]
    fn compile_plan_fallback_admits_off_device_candidate() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let mut table = KernelBindingTable::new();
        // Add f32 registered ONLY on Cpu — the pinned CUDA device
        // has no impl.
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );

        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let add = g.push(Node {
            op: Op::Add,
            inputs: vec![lhs, lhs],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let order = topo_order(&g, add);

        let placements_fn = move |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == cuda0 { vec![BackendId::Cuda] } else { vec![BackendId::Cpu] }
        };
        let fallback_fn = |_dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            vec![(BackendId::Cpu, DeviceLocation::Cpu)]
        };
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_pinned_device(cuda0)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add).expect("Add planned via fallback");
        assert_eq!(alts.len(), 1, "fallback set frozen to one winner");
        let w = alts.winner().unwrap();
        assert_eq!(w.backend, BackendId::Cpu);
        assert_eq!(w.device, DeviceLocation::Cpu, "off-device placement");
    }

    /// Step 4b locality, UNPRICED regime: without a Stage-2
    /// transfer estimator the fallback enumerator is NOT consulted
    /// while the pinned device has a registered candidate — an
    /// unpriced rank must never move an op cross-device on kernel
    /// cost alone.
    #[test]
    fn compile_plan_fallback_not_consulted_when_pinned_has_impl() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        register_add_f32(
            &mut table,
            BackendId::Cuda,
            noop_kernel_b,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu { vec![BackendId::Cpu] } else { vec![] }
        };
        let fallback_fn = |_dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            panic!(
                "fallback consulted although the pinned device has an \
                 implementation — locality policy violated",
            );
        };
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_pinned_device(DeviceLocation::Cpu)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add_id).unwrap();
        assert_eq!(alts.len(), 1);
        assert_eq!(alts.winner().unwrap().device, DeviceLocation::Cpu);
    }

    /// Step 4b: destructive ops never fall back — the plan-time
    /// NoBackendForOp error is preserved.
    #[test]
    fn compile_plan_fallback_denied_for_destructive_ops() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let mut table = KernelBindingTable::new();
        // ReluInplace registered ONLY on Cpu; pinned device is CUDA.
        table.register_full(
            OpKind::ReluInplace,
            &[DType::F32, DType::F32],
            BackendId::Cpu,
            noop_kernel,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            unknown_cost,
        );
        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let relu = g.push(Node {
            op: Op::ReluInplace,
            inputs: vec![lhs],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let order = topo_order(&g, relu);
        let placements_fn = move |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == cuda0 { vec![BackendId::Cuda] } else { vec![BackendId::Cpu] }
        };
        let fallback_fn = |_dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            vec![(BackendId::Cpu, DeviceLocation::Cpu)]
        };
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_pinned_device(cuda0)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn);
        let err = compile_plan(&g, &order, &table, &opts).unwrap_err();
        assert!(
            matches!(err, Error::NoBackendForOp { op: OpKind::ReluInplace, .. }),
            "destructive op must NOT fall back off-device; got {err:?}",
        );
    }

    /// Step 4b: even when MULTIPLE off-device backends could serve
    /// the op, the fallback set freezes to a single winner so the
    /// dispatch-time selector can't diverge from the residency
    /// stitch.
    #[test]
    fn compile_plan_fallback_freezes_to_single_winner() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let vk0 = DeviceLocation::Vulkan { gpu_id: 0 };
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        register_add_f32(
            &mut table,
            BackendId::Vulkan,
            noop_kernel_b,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );

        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let add = g.push(Node {
            op: Op::Add,
            inputs: vec![lhs, lhs],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let order = topo_order(&g, add);
        let placements_fn = move |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == cuda0 { vec![BackendId::Cuda] } else { vec![] }
        };
        let fallback_fn = move |_dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            vec![
                (BackendId::Cpu, DeviceLocation::Cpu),
                (BackendId::Vulkan, vk0),
            ]
        };
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_pinned_device(cuda0)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add).unwrap();
        assert_eq!(
            alts.len(),
            1,
            "fallback set must freeze to its single ranked winner",
        );
    }

    /// Synthetic Stage-2 estimator: zero same-device, flat latency
    /// otherwise. Unit tests must never depend on live calibration.
    struct FlatEstimator {
        latency_ns: u64,
    }

    impl crate::ranker::TransferEstimator for FlatEstimator {
        fn estimate_transfer_ns(
            &self,
            src: DeviceLocation,
            dst: DeviceLocation,
            _bytes: u64,
        ) -> u64 {
            if src == dst { 0 } else { self.latency_ns }
        }
    }

    /// Stage 2 residency threading: a missing-impl fallback set is
    /// priced against the inputs' residency — with both inputs
    /// resident on vk0, the Vulkan fallback candidate beats the
    /// first-enumerated CPU one (both have zero kernel cost, so the
    /// transfer term decides).
    #[test]
    fn compile_plan_prices_fallback_by_input_residency() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let vk0 = DeviceLocation::Vulkan { gpu_id: 0 };
        let mut table = KernelBindingTable::new();
        // No CUDA impl — the pinned cuda0 device must fall back.
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        register_add_f32(
            &mut table,
            BackendId::Vulkan,
            noop_kernel_b,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );

        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[4]),
            dtype: DType::F32,
        });
        let add = g.push(Node {
            op: Op::Add,
            inputs: vec![lhs, lhs],
            shape: Shape::from_dims(&[4]),
            dtype: DType::F32,
        });
        let order = topo_order(&g, add);

        let placements_fn = move |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == cuda0 { vec![BackendId::Cuda] } else { vec![] }
        };
        let fallback_fn = move |_dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            vec![
                (BackendId::Cpu, DeviceLocation::Cpu),
                (BackendId::Vulkan, vk0),
            ]
        };
        // The const's bytes live on vk0 (e.g. a persistent cache
        // slot) — exactly what the bridge's input-residency callback
        // reports.
        let residency_fn = move |id: NodeId| -> Option<DeviceLocation> {
            (id == lhs).then_some(vk0)
        };
        let est = FlatEstimator { latency_ns: 1_000 };
        let cpu_caps_val = cpu_caps();
        let caps_fn = |_: BackendId| Some(&cpu_caps_val);
        let opts = PlanOptions::new()
            .with_pinned_device(cuda0)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn)
            .with_capabilities_for(&caps_fn)
            .with_transfer_estimator(&est)
            .with_input_residency(&residency_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add).expect("Add planned via fallback");
        assert_eq!(alts.len(), 1, "fallback narrows to a single winner");
        let w = alts.winner().unwrap();
        assert_eq!(
            w.device, vk0,
            "transfer pricing moves the fallback to the inputs' residency",
        );
        assert_eq!(w.inbound_transfer_ns, 0, "co-resident inputs price zero");
    }

    /// Plan-time cost fns for the Stage-2 relax tests. Composite is
    /// the flops figure directly (no bytes, no overhead).
    fn cost_600(
        _: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities,
    ) -> crate::fused::CostEstimate {
        crate::fused::CostEstimate { flops: 600, bytes_moved: 0, kernel_overhead_ns: 0 }
    }
    fn cost_10(
        _: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities,
    ) -> crate::fused::CostEstimate {
        crate::fused::CostEstimate { flops: 10, bytes_moved: 0, kernel_overhead_ns: 0 }
    }
    fn cost_10_000_000(
        _: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities,
    ) -> crate::fused::CostEstimate {
        crate::fused::CostEstimate {
            flops: 10_000_000,
            bytes_moved: 0,
            kernel_overhead_ns: 0,
        }
    }

    /// Test fixture for the relax: Add f32 on CPU (cost via
    /// `cpu_cost`) + CUDA at cuda:0 (cost via `cuda_cost`), graph
    /// pinned to CPU, both consts resident on CPU, fallback
    /// enumerator offering cuda:0. Returns the planned set for the
    /// Add node.
    fn relaxed_plan_with_costs(
        cpu_cost: crate::kernel::CostFn,
        cuda_cost: crate::kernel::CostFn,
        estimator: &FlatEstimator,
    ) -> AlternativeSet {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let mut table = KernelBindingTable::new();
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            noop_kernel,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            cpu_cost,
        );
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cuda,
            noop_kernel_b,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            cuda_cost,
        );

        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu { vec![BackendId::Cpu] } else { vec![] }
        };
        let fallback_fn = move |_dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            vec![(BackendId::Cuda, cuda0)]
        };
        // build_add_graph's consts have no cache entry; report them
        // CPU-resident as build_const_cache would for a CPU realize.
        let residency_fn = |_id: NodeId| -> Option<DeviceLocation> {
            Some(DeviceLocation::Cpu)
        };
        let cpu_caps_val = cpu_caps();
        let caps_fn = |_: BackendId| Some(&cpu_caps_val);
        let opts = PlanOptions::new()
            .with_pinned_device(DeviceLocation::Cpu)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn)
            .with_capabilities_for(&caps_fn)
            .with_transfer_estimator(estimator)
            .with_input_residency(&residency_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        plan.alternatives(add_id).expect("Add planned").clone()
    }

    /// Stage-2 parity gate: a tiny op stays local although the
    /// remote kernel is "faster" — the transfer term dominates.
    /// Off-device siblings are pruned from the surviving set.
    #[test]
    fn compile_plan_relaxed_tiny_op_stays_local_when_transfer_dominates() {
        // CPU 600 ns vs CUDA 10 ns kernel; 1 ms per input crossing.
        // CUDA total = 10 + 2 × 1_000_000 ≫ CPU 600.
        let est = FlatEstimator { latency_ns: 1_000_000 };
        let set = relaxed_plan_with_costs(cost_600, cost_10, &est);
        let w = set.winner().unwrap();
        assert_eq!(w.device, DeviceLocation::Cpu, "transfer dominates → stay local");
        assert_eq!(w.inbound_transfer_ns, 0, "local winner pays no transfer");
        assert!(
            set.alternatives().iter().all(|c| c.device == DeviceLocation::Cpu),
            "off-device siblings pruned after rank",
        );
    }

    /// Stage-2 parity gate: a huge op legitimately moves — the
    /// kernel gap dwarfs the transfer. The surviving set lives
    /// entirely on the winner's device so dispatch-time selectors
    /// can't diverge from the residency stitch.
    #[test]
    fn compile_plan_relaxed_huge_op_moves_when_kernel_gap_dominates() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        // CPU 10 ms vs CUDA 10 ns kernel; 1 µs per input crossing.
        let est = FlatEstimator { latency_ns: 1_000 };
        let set = relaxed_plan_with_costs(cost_10_000_000, cost_10, &est);
        let w = set.winner().unwrap();
        assert_eq!(w.device, cuda0, "kernel gap dominates → move");
        assert_eq!(
            w.inbound_transfer_ns,
            2_000,
            "the move's two input crossings are priced on the winner",
        );
        assert!(
            set.alternatives().iter().all(|c| c.device == cuda0),
            "set pruned to the winner's device",
        );
    }

    /// Stage-2 parity gate: zero signal everywhere (zero kernel
    /// costs, zero-cost transfers) must still keep the op local —
    /// the decision device enumerates first and the rank's stable
    /// sort preserves that order on ties.
    #[test]
    fn compile_plan_relaxed_zero_signal_ties_stay_local() {
        let est = FlatEstimator { latency_ns: 0 };
        let set = relaxed_plan_with_costs(unknown_cost, unknown_cost, &est);
        assert_eq!(
            set.winner().unwrap().device,
            DeviceLocation::Cpu,
            "ties break toward the decision device",
        );
    }

    /// Stage-2 parity gate: single-device systems produce
    /// byte-identical plans with the estimator wired — the fallback
    /// enumerator has nothing to offer and no transfer term fires.
    #[test]
    fn compile_plan_relaxed_single_device_plan_unchanged() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu { vec![BackendId::Cpu] } else { vec![] }
        };
        // Single-device system: no other device exists.
        let fallback_fn =
            |_dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> { Vec::new() };
        let residency_fn =
            |_id: NodeId| -> Option<DeviceLocation> { Some(DeviceLocation::Cpu) };
        let cpu_caps_val = cpu_caps();
        let caps_fn = |_: BackendId| Some(&cpu_caps_val);

        let base_opts = PlanOptions::new()
            .with_pinned_device(DeviceLocation::Cpu)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn)
            .with_capabilities_for(&caps_fn);
        let base = compile_plan(&g, &order, &table, &base_opts).expect("base");

        let est = FlatEstimator { latency_ns: 1_000_000 };
        let wired_opts = PlanOptions::new()
            .with_pinned_device(DeviceLocation::Cpu)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn)
            .with_capabilities_for(&caps_fn)
            .with_transfer_estimator(&est)
            .with_input_residency(&residency_fn);
        let wired = compile_plan(&g, &order, &table, &wired_opts).expect("wired");

        let a = base.alternatives(add_id).unwrap();
        let b = wired.alternatives(add_id).unwrap();
        assert_eq!(a.len(), b.len(), "same candidate count");
        for (ca, cb) in a.alternatives().iter().zip(b.alternatives()) {
            assert_eq!(ca.backend, cb.backend);
            assert_eq!(ca.device, cb.device);
            assert_eq!(ca.kernel as usize, cb.kernel as usize, "same kernel ref");
            assert_eq!(ca.static_cost, cb.static_cost, "same Layer-1 cost");
            assert_eq!(cb.inbound_transfer_ns, 0, "no transfer term fires");
        }
        assert_eq!(a.context(), b.context(), "same decision context");
    }

    /// Stage-2 hard pin: an explicit `Graph::placement` keeps the
    /// node on its device even with pricing active and a free
    /// "faster" remote sibling — the fallback enumerator is never
    /// consulted for the pinned node.
    #[test]
    fn compile_plan_relaxed_respects_explicit_placement_pin() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        register_add_f32(
            &mut table,
            BackendId::Cuda,
            noop_kernel_b,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        let (mut g, add_id) = build_add_graph();
        // Hard pin: scheduler / Judge-profiling decision.
        g.set_placement(add_id, DeviceLocation::Cpu);
        let order = topo_order(&g, add_id);
        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu { vec![BackendId::Cpu] } else { vec![] }
        };
        let fallback_fn = |_dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            panic!(
                "fallback consulted for a hard-pinned node — explicit \
                 placements must not enter the priced relax",
            );
        };
        let residency_fn =
            |_id: NodeId| -> Option<DeviceLocation> { Some(DeviceLocation::Cpu) };
        let est = FlatEstimator { latency_ns: 0 };
        let cpu_caps_val = cpu_caps();
        let caps_fn = |_: BackendId| Some(&cpu_caps_val);
        let opts = PlanOptions::new()
            .with_pinned_device(DeviceLocation::Cpu)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn)
            .with_capabilities_for(&caps_fn)
            .with_transfer_estimator(&est)
            .with_input_residency(&residency_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        assert_eq!(
            plan.alternatives(add_id).unwrap().winner().unwrap().device,
            DeviceLocation::Cpu,
        );
    }

    /// Stage-2 hard pin: destructive ops never enter the priced
    /// relax — a free, faster off-device sibling must not win.
    #[test]
    fn compile_plan_relaxed_destructive_never_moves() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let mut table = KernelBindingTable::new();
        table.register_full(
            OpKind::ReluInplace,
            &[DType::F32, DType::F32],
            BackendId::Cpu,
            noop_kernel,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            cost_600,
        );
        table.register_full(
            OpKind::ReluInplace,
            &[DType::F32, DType::F32],
            BackendId::Cuda,
            noop_kernel_b,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            cost_10,
        );
        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let relu = g.push(Node {
            op: Op::ReluInplace,
            inputs: vec![lhs],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let order = topo_order(&g, relu);
        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu { vec![BackendId::Cpu] } else { vec![] }
        };
        let fallback_fn = move |_dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            vec![(BackendId::Cuda, cuda0)]
        };
        let residency_fn =
            |_id: NodeId| -> Option<DeviceLocation> { Some(DeviceLocation::Cpu) };
        let est = FlatEstimator { latency_ns: 0 };
        let cpu_caps_val = cpu_caps();
        let caps_fn = |_: BackendId| Some(&cpu_caps_val);
        let opts = PlanOptions::new()
            .with_pinned_device(DeviceLocation::Cpu)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn)
            .with_capabilities_for(&caps_fn)
            .with_transfer_estimator(&est)
            .with_input_residency(&residency_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        assert_eq!(
            plan.alternatives(relu).unwrap().winner().unwrap().device,
            DeviceLocation::Cpu,
            "destructive op stays on the device owning its mutation target",
        );
    }

    /// Stage-2 fix of the picker-4b verifier minor: a pinned device
    /// whose registrations all fail the HARD filter chain falls
    /// back off-device like an empty enumeration (legacy/unpriced
    /// regime — no estimator configured).
    #[test]
    fn compile_plan_filter_rejected_pinned_device_falls_back() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let mut table = KernelBindingTable::new();
        // The pinned CUDA device's only registration is NOT
        // bit-stable; the CPU sibling is.
        register_add_f32(
            &mut table,
            BackendId::Cuda,
            noop_kernel_b,
            PrecisionGuarantee {
                bit_stable_on_same_hardware: false,
                ..PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU
            },
        );
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let add = g.push(Node {
            op: Op::Add,
            inputs: vec![lhs, lhs],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let order = topo_order(&g, add);
        let placements_fn = move |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == cuda0 { vec![BackendId::Cuda] } else { vec![] }
        };
        let fallback_fn = |_dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            vec![(BackendId::Cpu, DeviceLocation::Cpu)]
        };
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_pinned_device(cuda0)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn)
            .with_precision_requirement(PrecisionRequirement::BIT_STABLE);
        let plan = compile_plan(&g, &order, &table, &opts)
            .expect("filter-rejected pin falls back instead of erroring");
        let alts = plan.alternatives(add).expect("Add planned via fallback");
        assert_eq!(alts.len(), 1, "fallback set frozen to one winner");
        let w = alts.winner().unwrap();
        assert_eq!(w.backend, BackendId::Cpu);
        assert_eq!(w.device, DeviceLocation::Cpu);
        assert!(
            w.precision.bit_stable_on_same_hardware,
            "the admissible off-device sibling won",
        );
    }

    /// Stage 2: without an estimator nothing prices — candidates
    /// keep `inbound_transfer_ns == 0` and the plan matches the
    /// estimator-less plan candidate-for-candidate even when an
    /// input-residency callback is supplied.
    #[test]
    fn compile_plan_without_estimator_prices_nothing() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let cpu_caps_val = cpu_caps();
        let caps_fn = |_: BackendId| Some(&cpu_caps_val);
        let residency_fn =
            |_id: NodeId| -> Option<DeviceLocation> { Some(DeviceLocation::Cpu) };

        let base_opts = PlanOptions::new().with_capabilities_for(&caps_fn);
        let base = compile_plan(&g, &order, &table, &base_opts).expect("base");

        let with_residency = PlanOptions::new()
            .with_capabilities_for(&caps_fn)
            .with_input_residency(&residency_fn);
        let plan = compile_plan(&g, &order, &table, &with_residency).expect("plan");

        let a = base.alternatives(add_id).unwrap();
        let b = plan.alternatives(add_id).unwrap();
        assert_eq!(a.len(), b.len());
        for (ca, cb) in a.alternatives().iter().zip(b.alternatives()) {
            assert_eq!(ca.backend, cb.backend);
            assert_eq!(ca.device, cb.device);
            assert_eq!(ca.kernel as usize, cb.kernel as usize);
            assert_eq!(ca.inbound_transfer_ns, 0);
            assert_eq!(cb.inbound_transfer_ns, 0);
        }
    }

    #[test]
    fn compile_plan_passes_input_layouts_to_filter_chain() {
        // Smoke test: a non-contiguous input + strided-input pref
        // in the default chain prefers the strided-capable kernel.
        // Register two CPU kernels, one with strided caps.
        let mut table = KernelBindingTable::new();
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            noop_kernel,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            unknown_cost,
        );
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            noop_kernel_b,
            KernelCaps::strided_input(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            unknown_cost,
        );

        // Build a graph where the Add's LHS input is non-contiguous.
        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[6]),
            dtype: DType::F32,
        });
        // Force a non-contiguous layout on lhs by setting a custom
        // stride via the graph's layout side-table.
        let strides: StrideVec = smallvec![2isize];
        g.set_layout(lhs, Layout::new(Shape::from_dims(&[6]), strides, 0));
        let rhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[6]),
            dtype: DType::F32,
        });
        let add = g.push(Node {
            op: Op::Add,
            inputs: vec![lhs, rhs],
            shape: Shape::from_dims(&[6]),
            dtype: DType::F32,
        });
        g.set_target_backend(add, BackendId::Cpu);
        let order = topo_order(&g, add);
        let opts = PlanOptions::new().without_cost_population();
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add).unwrap();
        // Strided-input preference narrowed to the strided-capable
        // kernel.
        assert_eq!(alts.len(), 1);
        assert!(alts.winner().unwrap().caps.strided_input);
    }

    // =====================================================================
    // Planner Stage 3: carry-forward placement DP
    // =====================================================================

    /// Byte-aware synthetic estimator: zero same-device, otherwise
    /// `latency + bytes·ns_per_byte`.
    struct BytesEstimator {
        latency_ns: u64,
        ns_per_byte: u64,
    }

    impl crate::ranker::TransferEstimator for BytesEstimator {
        fn estimate_transfer_ns(
            &self,
            src: DeviceLocation,
            dst: DeviceLocation,
            bytes: u64,
        ) -> u64 {
            if src == dst {
                return 0;
            }
            self.latency_ns
                .saturating_add(bytes.saturating_mul(self.ns_per_byte))
        }
    }

    fn cost_50(
        _: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities,
    ) -> crate::fused::CostEstimate {
        crate::fused::CostEstimate { flops: 50, bytes_moved: 0, kernel_overhead_ns: 0 }
    }
    fn cost_100(
        _: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities,
    ) -> crate::fused::CostEstimate {
        crate::fused::CostEstimate { flops: 100, bytes_moved: 0, kernel_overhead_ns: 0 }
    }
    fn cost_300(
        _: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities,
    ) -> crate::fused::CostEstimate {
        crate::fused::CostEstimate { flops: 300, bytes_moved: 0, kernel_overhead_ns: 0 }
    }
    fn cost_800(
        _: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities,
    ) -> crate::fused::CostEstimate {
        crate::fused::CostEstimate { flops: 800, bytes_moved: 0, kernel_overhead_ns: 0 }
    }
    fn cost_5000(
        _: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities,
    ) -> crate::fused::CostEstimate {
        crate::fused::CostEstimate { flops: 5000, bytes_moved: 0, kernel_overhead_ns: 0 }
    }
    fn cost_100_000(
        _: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities,
    ) -> crate::fused::CostEstimate {
        crate::fused::CostEstimate {
            flops: 100_000,
            bytes_moved: 0,
            kernel_overhead_ns: 0,
        }
    }

    /// Register a unary `(F32 → F32)` kernel with an explicit cost.
    fn register_unary_f32(
        table: &mut KernelBindingTable,
        op: OpKind,
        backend: BackendId,
        kernel: crate::kernel::KernelRef,
        cost: crate::kernel::CostFn,
    ) {
        table.register_full(
            op,
            &[DType::F32, DType::F32],
            backend,
            kernel,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            cost,
        );
    }

    fn push_f32(g: &mut Graph, op: Op, inputs: Vec<NodeId>, dims: &[usize]) -> NodeId {
        g.push(Node {
            op,
            inputs,
            shape: Shape::from_dims(dims),
            dtype: DType::F32,
        })
    }

    /// Stage 3 test (a): a mid-sequence GPU segment beats all-CPU
    /// and all-GPU, and the per-op GPU win (700 ns) is SMALLER than
    /// the crossing (1000 ns) — greedy would never move (the first
    /// op of the migration can't justify the crossing alone), but
    /// the accumulated DP state amortizes one entry + one exit
    /// crossing over the segment's combined win.
    ///
    /// Chain: c → n1(Sqr) → n2(Neg) → n3(Neg) → n4(Neg) → n5(Sqr).
    /// Sqr: CPU 100 / GPU 5000 (GPU-hostile endpoints).
    /// Neg: CPU 800 / GPU 100 (per-op win 700 < 1000 crossing).
    ///
    /// All-CPU = 100 + 3·800 + 100 = 2600.
    /// Mid-GPU = 100 + 1000 + 3·100 + 1000 + 100 = 2500. ← winner
    /// All-GPU = 1000 + 5000 + 3·100 + 5000 + 1000 = 12300.
    #[test]
    fn dp_mid_sequence_gpu_segment_beats_all_cpu_and_all_gpu() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let mut table = KernelBindingTable::new();
        register_unary_f32(&mut table, OpKind::SqrElementwise, BackendId::Cpu, noop_kernel, cost_100);
        register_unary_f32(&mut table, OpKind::SqrElementwise, BackendId::Cuda, noop_kernel_b, cost_5000);
        register_unary_f32(&mut table, OpKind::NegElementwise, BackendId::Cpu, noop_kernel, cost_800);
        register_unary_f32(&mut table, OpKind::NegElementwise, BackendId::Cuda, noop_kernel_b, cost_100);

        let mut g = Graph::new();
        let c = push_f32(&mut g, Op::Const, vec![], &[4]);
        let n1 = push_f32(&mut g, Op::Sqr, vec![c], &[4]);
        let n2 = push_f32(&mut g, Op::Neg, vec![n1], &[4]);
        let n3 = push_f32(&mut g, Op::Neg, vec![n2], &[4]);
        let n4 = push_f32(&mut g, Op::Neg, vec![n3], &[4]);
        let n5 = push_f32(&mut g, Op::Sqr, vec![n4], &[4]);
        let order = topo_order(&g, n5);

        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu { vec![BackendId::Cpu] } else { vec![BackendId::Cuda] }
        };
        let fallback_fn = move |dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            if dev == DeviceLocation::Cpu {
                vec![(BackendId::Cuda, cuda0)]
            } else {
                vec![(BackendId::Cpu, DeviceLocation::Cpu)]
            }
        };
        let residency_fn =
            move |id: NodeId| -> Option<DeviceLocation> { (id == c).then_some(DeviceLocation::Cpu) };
        let est = FlatEstimator { latency_ns: 1_000 };
        let cpu_caps_val = cpu_caps();
        let caps_fn = |_: BackendId| Some(&cpu_caps_val);
        let opts = PlanOptions::new()
            .with_pinned_device(DeviceLocation::Cpu)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn)
            .with_capabilities_for(&caps_fn)
            .with_transfer_estimator(&est)
            .with_input_residency(&residency_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");

        let dev_of = |id: NodeId| plan.alternatives(id).unwrap().winner().unwrap().device;
        assert_eq!(dev_of(n1), DeviceLocation::Cpu, "GPU-hostile entry stays CPU");
        assert_eq!(dev_of(n2), cuda0, "segment start migrates");
        assert_eq!(dev_of(n3), cuda0, "segment interior stays GPU");
        assert_eq!(dev_of(n4), cuda0, "segment end stays GPU");
        assert_eq!(dev_of(n5), DeviceLocation::Cpu, "GPU-hostile exit returns");

        // Transfer diagnostics on the final sets: the two crossings
        // are priced exactly once each, on the segment boundary
        // winners.
        let inbound = |id: NodeId| plan.alternatives(id).unwrap().winner().unwrap().inbound_transfer_ns;
        assert_eq!(inbound(n2), 1_000, "entry crossing priced on the first GPU op");
        assert_eq!(inbound(n3), 0);
        assert_eq!(inbound(n4), 0);
        assert_eq!(inbound(n5), 1_000, "exit crossing priced on the return op");

        // The surviving sets live on ONE device each (residency
        // stitch invariant).
        for id in [n1, n2, n3, n4, n5] {
            let d = dev_of(id);
            assert!(
                plan.alternatives(id).unwrap().alternatives().iter().all(|c| c.device == d),
                "set pruned to the DP-committed device",
            );
        }
    }

    /// Stage 3 test (b): a FUSED op that is locally faster on device
    /// A loses to staying on B when its stranding cost exceeds its
    /// win — the over-move case greedy inbound pricing cannot see.
    ///
    /// Chain: c → n1(Neg) → f(Fused SoftmaxLastDim) → n2(Sqr, CPU-only).
    /// Estimator: 10 ns latency + 1 ns/byte. n1 output is 4 bytes;
    /// f's output is 1024 bytes (shapes are synthetic — compile_plan
    /// never executes).
    ///
    /// n1: CPU 100 000 / GPU 100 → migrates to GPU regardless.
    /// f:  CPU 300 / GPU 100. Arriving states: CPU = 300 + 14 = 428
    /// (via GPU n1), GPU = 100 + 114 = 214 — locally GPU wins, and
    /// greedy would commit it (inbound favors GPU: 100 + 0 < 300 +
    /// 14). But f's 1024-byte output must land on CPU for n2:
    /// GPU 214 + 1034 = 1248 vs CPU 428 → the DP keeps f on CPU.
    #[test]
    fn dp_fused_op_loses_when_stranding_exceeds_local_win() {
        use fuel_graph::registry::{FusedOpParams, FusedOps};

        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let mut table = KernelBindingTable::new();
        register_unary_f32(&mut table, OpKind::NegElementwise, BackendId::Cpu, noop_kernel, cost_100_000);
        register_unary_f32(&mut table, OpKind::NegElementwise, BackendId::Cuda, noop_kernel_b, cost_100);
        register_unary_f32(&mut table, OpKind::SoftmaxLastDim, BackendId::Cpu, noop_kernel, cost_300);
        register_unary_f32(&mut table, OpKind::SoftmaxLastDim, BackendId::Cuda, noop_kernel_b, cost_100);
        // Sqr exists ONLY on CPU — the chain's anchor.
        register_unary_f32(&mut table, OpKind::SqrElementwise, BackendId::Cpu, noop_kernel, cost_100);

        let mut g = Graph::new();
        let c = push_f32(&mut g, Op::Const, vec![], &[1]);
        let n1 = push_f32(&mut g, Op::Neg, vec![c], &[1]);
        let f = push_f32(
            &mut g,
            Op::Fused(FusedOps::SOFTMAX_LAST_DIM, FusedOpParams::SoftmaxLastDim),
            vec![n1],
            &[256],
        );
        let n2 = push_f32(&mut g, Op::Sqr, vec![f], &[256]);
        let order = topo_order(&g, n2);

        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu { vec![BackendId::Cpu] } else { vec![BackendId::Cuda] }
        };
        let fallback_fn = move |dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            if dev == DeviceLocation::Cpu {
                vec![(BackendId::Cuda, cuda0)]
            } else {
                vec![(BackendId::Cpu, DeviceLocation::Cpu)]
            }
        };
        let residency_fn =
            move |id: NodeId| -> Option<DeviceLocation> { (id == c).then_some(DeviceLocation::Cpu) };
        let est = BytesEstimator { latency_ns: 10, ns_per_byte: 1 };
        let cpu_caps_val = cpu_caps();
        let caps_fn = |_: BackendId| Some(&cpu_caps_val);
        let opts = PlanOptions::new()
            .with_pinned_device(DeviceLocation::Cpu)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn)
            .with_capabilities_for(&caps_fn)
            .with_transfer_estimator(&est)
            .with_input_residency(&residency_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");

        let dev_of = |id: NodeId| plan.alternatives(id).unwrap().winner().unwrap().device;
        assert_eq!(dev_of(n1), cuda0, "the huge-win producer migrates");
        assert_eq!(
            dev_of(f),
            DeviceLocation::Cpu,
            "the locally-faster fused op stays where its output is needed",
        );
        assert_eq!(dev_of(n2), DeviceLocation::Cpu);
        assert!(
            plan.alternatives(f).unwrap().alternatives().iter().all(|c| c.device == DeviceLocation::Cpu),
            "fused set pruned to the committed device",
        );
    }

    /// Stage 3 test (c): exit pricing — a final op stays on the
    /// realize target although moving would win locally, because
    /// the return crossing prices the move out.
    ///
    /// Inputs resident on GPU; realize target CPU; flat 1000 ns
    /// crossings. Local view: GPU 100 + 0 ≪ CPU 50 + 1000. With the
    /// exit: GPU 100 + 1000 = 1100 vs CPU 1050 → CPU.
    #[test]
    fn dp_exit_pricing_keeps_final_op_on_realize_target() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let mut table = KernelBindingTable::new();
        register_unary_f32(&mut table, OpKind::NegElementwise, BackendId::Cpu, noop_kernel, cost_50);
        register_unary_f32(&mut table, OpKind::NegElementwise, BackendId::Cuda, noop_kernel_b, cost_100);

        let mut g = Graph::new();
        let c = push_f32(&mut g, Op::Const, vec![], &[4]);
        let n = push_f32(&mut g, Op::Neg, vec![c], &[4]);
        let order = topo_order(&g, n);

        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu { vec![BackendId::Cpu] } else { vec![BackendId::Cuda] }
        };
        let fallback_fn = move |dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            if dev == DeviceLocation::Cpu {
                vec![(BackendId::Cuda, cuda0)]
            } else {
                vec![(BackendId::Cpu, DeviceLocation::Cpu)]
            }
        };
        let residency_fn =
            move |id: NodeId| -> Option<DeviceLocation> { (id == c).then_some(cuda0) };
        let est = FlatEstimator { latency_ns: 1_000 };
        let cpu_caps_val = cpu_caps();
        let caps_fn = |_: BackendId| Some(&cpu_caps_val);
        let opts = PlanOptions::new()
            .with_pinned_device(DeviceLocation::Cpu)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn)
            .with_capabilities_for(&caps_fn)
            .with_transfer_estimator(&est)
            .with_input_residency(&residency_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let w = plan.alternatives(n).unwrap().winner().unwrap();
        assert_eq!(
            w.device,
            DeviceLocation::Cpu,
            "exit pricing flips the locally-winning GPU placement",
        );
        assert_eq!(
            w.inbound_transfer_ns, 1_000,
            "the GPU-resident input's crossing is priced on the CPU winner",
        );
    }

    /// Stage 3 test (c) inverse: when the kernel gap dwarfs both
    /// crossings, the final op legitimately moves despite the exit.
    #[test]
    fn dp_exit_pricing_does_not_pin_when_kernel_gap_dominates() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let mut table = KernelBindingTable::new();
        register_unary_f32(&mut table, OpKind::NegElementwise, BackendId::Cpu, noop_kernel, cost_10_000_000);
        register_unary_f32(&mut table, OpKind::NegElementwise, BackendId::Cuda, noop_kernel_b, cost_100);

        let mut g = Graph::new();
        let c = push_f32(&mut g, Op::Const, vec![], &[4]);
        let n = push_f32(&mut g, Op::Neg, vec![c], &[4]);
        let order = topo_order(&g, n);

        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu { vec![BackendId::Cpu] } else { vec![BackendId::Cuda] }
        };
        let fallback_fn = move |dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            if dev == DeviceLocation::Cpu {
                vec![(BackendId::Cuda, cuda0)]
            } else {
                vec![(BackendId::Cpu, DeviceLocation::Cpu)]
            }
        };
        let residency_fn =
            move |id: NodeId| -> Option<DeviceLocation> { (id == c).then_some(cuda0) };
        let est = FlatEstimator { latency_ns: 1_000 };
        let cpu_caps_val = cpu_caps();
        let caps_fn = |_: BackendId| Some(&cpu_caps_val);
        let opts = PlanOptions::new()
            .with_pinned_device(DeviceLocation::Cpu)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn)
            .with_capabilities_for(&caps_fn)
            .with_transfer_estimator(&est)
            .with_input_residency(&residency_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        assert_eq!(plan.alternatives(n).unwrap().winner().unwrap().device, cuda0);
    }

    /// Stage 3 test (d): single-DEVICE plan equality. Two backends
    /// co-located on one device with the full Stage-3 option set
    /// wired (estimator + fallback + residency) produce the same
    /// plan, candidate-for-candidate, as the unwired baseline — the
    /// DP never opens a row when only one device has candidates.
    #[test]
    fn dp_single_device_plan_equals_greedy_plan() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        register_add_f32(
            &mut table,
            BackendId::Cuda,
            noop_kernel_b,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        // Both backends co-located at DeviceLocation::Cpu — one
        // device, two backends.
        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu {
                vec![BackendId::Cpu, BackendId::Cuda]
            } else {
                vec![]
            }
        };
        let fallback_fn =
            |_dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> { Vec::new() };
        let residency_fn =
            |_id: NodeId| -> Option<DeviceLocation> { Some(DeviceLocation::Cpu) };
        let cpu_caps_val = cpu_caps();
        let caps_fn = |_: BackendId| Some(&cpu_caps_val);

        let base_opts = PlanOptions::new()
            .with_placements_for_device(&placements_fn)
            .with_capabilities_for(&caps_fn);
        let base = compile_plan(&g, &order, &table, &base_opts).expect("base");

        let est = FlatEstimator { latency_ns: 1_000_000 };
        let wired_opts = PlanOptions::new()
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn)
            .with_capabilities_for(&caps_fn)
            .with_transfer_estimator(&est)
            .with_input_residency(&residency_fn);
        let wired = compile_plan(&g, &order, &table, &wired_opts).expect("wired");

        assert_eq!(base.alternatives.len(), wired.alternatives.len());
        let a = base.alternatives(add_id).unwrap();
        let b = wired.alternatives(add_id).unwrap();
        assert_eq!(a.len(), b.len(), "same candidate count");
        for (ca, cb) in a.alternatives().iter().zip(b.alternatives()) {
            assert_eq!(ca.backend, cb.backend);
            assert_eq!(ca.device, cb.device);
            assert_eq!(ca.kernel as usize, cb.kernel as usize, "same kernel ref");
            assert_eq!(ca.static_cost, cb.static_cost, "same Layer-1 cost");
            assert_eq!(cb.inbound_transfer_ns, 0, "no transfer term fires");
        }
        assert_eq!(a.context(), b.context(), "same decision context");
    }

    /// Stage 3 test (e) helper: build the diamond (a → b, a → c2,
    /// b+c2 → d), plan it with the given crossing latency, and
    /// assert every kernel-bearing node committed to `expect`.
    fn run_diamond_regime(latency_ns: u64, expect: DeviceLocation) {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let mut table = KernelBindingTable::new();
        for op in [
            OpKind::NegElementwise,
            OpKind::SqrElementwise,
            OpKind::ReluElementwise,
        ] {
            register_unary_f32(&mut table, op, BackendId::Cpu, noop_kernel, cost_100);
            register_unary_f32(&mut table, op, BackendId::Cuda, noop_kernel_b, cost_10);
        }
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            noop_kernel,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            cost_100,
        );
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cuda,
            noop_kernel_b,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            cost_10,
        );

        let mut g = Graph::new();
        let c0 = push_f32(&mut g, Op::Const, vec![], &[4]);
        let a = push_f32(&mut g, Op::Neg, vec![c0], &[4]);
        let b = push_f32(&mut g, Op::Sqr, vec![a], &[4]);
        let c2 = push_f32(&mut g, Op::Relu, vec![a], &[4]);
        let d = push_f32(&mut g, Op::Add, vec![b, c2], &[4]);
        let order = topo_order(&g, d);

        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu { vec![BackendId::Cpu] } else { vec![BackendId::Cuda] }
        };
        let fallback_fn = move |dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            if dev == DeviceLocation::Cpu {
                vec![(BackendId::Cuda, cuda0)]
            } else {
                vec![(BackendId::Cpu, DeviceLocation::Cpu)]
            }
        };
        let residency_fn = move |id: NodeId| -> Option<DeviceLocation> {
            (id == c0).then_some(DeviceLocation::Cpu)
        };
        let est = FlatEstimator { latency_ns };
        let cpu_caps_val = cpu_caps();
        let caps_fn = |_: BackendId| Some(&cpu_caps_val);
        let opts = PlanOptions::new()
            .with_pinned_device(DeviceLocation::Cpu)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn)
            .with_capabilities_for(&caps_fn)
            .with_transfer_estimator(&est)
            .with_input_residency(&residency_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        for id in [a, b, c2, d] {
            assert_eq!(
                plan.alternatives(id).unwrap().winner().unwrap().device,
                expect,
                "diamond join must commit a consistent placement",
            );
        }
    }

    /// Stage 3 test (e): a diamond merges without panicking and
    /// commits a consistent placement. Both cost regimes exercised:
    /// transfers dominate → everything stays on CPU; kernel gap
    /// dominates with free transfers → everything moves to the GPU.
    #[test]
    fn dp_diamond_join_merges_consistently() {
        // 1 ms crossings — the 90 ns/op GPU win never pays.
        run_diamond_regime(1_000_000, DeviceLocation::Cpu);
        // Free transfers + 10× kernel gap.
        run_diamond_regime(0, DeviceLocation::Cuda { gpu_id: 0 });
    }

    /// Stage 3 perf guard: the DP is O(nodes × devices²). A
    /// 1000-node chain with two devices must plan comfortably under
    /// a wall-clock sanity bound (expected: low milliseconds — the
    /// generous bound only guards against accidental quadratic
    /// blowups in nodes).
    #[test]
    fn dp_thousand_node_chain_plans_within_sanity_bound() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let mut table = KernelBindingTable::new();
        // Equal costs — ties must break toward the decision device
        // along the whole chain.
        register_unary_f32(&mut table, OpKind::NegElementwise, BackendId::Cpu, noop_kernel, cost_100);
        register_unary_f32(&mut table, OpKind::NegElementwise, BackendId::Cuda, noop_kernel_b, cost_100);

        let mut g = Graph::new();
        let c = push_f32(&mut g, Op::Const, vec![], &[4]);
        let mut prev = c;
        let mut nodes = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let n = push_f32(&mut g, Op::Neg, vec![prev], &[4]);
            nodes.push(n);
            prev = n;
        }
        let order = topo_order(&g, prev);

        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu { vec![BackendId::Cpu] } else { vec![BackendId::Cuda] }
        };
        let fallback_fn = move |dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            if dev == DeviceLocation::Cpu {
                vec![(BackendId::Cuda, cuda0)]
            } else {
                vec![(BackendId::Cpu, DeviceLocation::Cpu)]
            }
        };
        let residency_fn =
            move |id: NodeId| -> Option<DeviceLocation> { (id == c).then_some(DeviceLocation::Cpu) };
        let est = FlatEstimator { latency_ns: 10 };
        let cpu_caps_val = cpu_caps();
        let caps_fn = |_: BackendId| Some(&cpu_caps_val);
        let opts = PlanOptions::new()
            .with_pinned_device(DeviceLocation::Cpu)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn)
            .with_capabilities_for(&caps_fn)
            .with_transfer_estimator(&est)
            .with_input_residency(&residency_fn);

        let start = std::time::Instant::now();
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs() < 5,
            "1000-node DP plan took {elapsed:?} — pathological scaling",
        );

        assert_eq!(plan.alternatives.len(), 1000);
        for &n in &nodes {
            assert_eq!(
                plan.alternatives(n).unwrap().winner().unwrap().device,
                DeviceLocation::Cpu,
                "equal-cost ties break toward the decision device end-to-end",
            );
        }
    }

    /// Stage 3: a view-shaped pass-through between two DP nodes
    /// keeps the chain connected (the alias map) — the segment still
    /// migrates as one unit instead of breaking at the Reshape.
    #[test]
    fn dp_chain_connects_through_view_passthroughs() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let mut table = KernelBindingTable::new();
        // Per-op win 700 < 1000 crossing; combined win 2100 > 2000 —
        // the segment only migrates if the chain survives the
        // Reshape between n1 and n2.
        register_unary_f32(&mut table, OpKind::NegElementwise, BackendId::Cpu, noop_kernel, cost_800);
        register_unary_f32(&mut table, OpKind::NegElementwise, BackendId::Cuda, noop_kernel_b, cost_100);

        let mut g = Graph::new();
        let c = push_f32(&mut g, Op::Const, vec![], &[4]);
        let n1 = push_f32(&mut g, Op::Neg, vec![c], &[4]);
        let r = push_f32(&mut g, Op::Reshape(Shape::from_dims(&[2, 2])), vec![n1], &[2, 2]);
        let n2 = push_f32(&mut g, Op::Neg, vec![r], &[2, 2]);
        let n3 = push_f32(&mut g, Op::Neg, vec![n2], &[2, 2]);
        let order = topo_order(&g, n3);

        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu { vec![BackendId::Cpu] } else { vec![BackendId::Cuda] }
        };
        let fallback_fn = move |dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            if dev == DeviceLocation::Cpu {
                vec![(BackendId::Cuda, cuda0)]
            } else {
                vec![(BackendId::Cpu, DeviceLocation::Cpu)]
            }
        };
        let residency_fn =
            move |id: NodeId| -> Option<DeviceLocation> { (id == c).then_some(DeviceLocation::Cpu) };
        let est = FlatEstimator { latency_ns: 1_000 };
        let cpu_caps_val = cpu_caps();
        let caps_fn = |_: BackendId| Some(&cpu_caps_val);
        let opts = PlanOptions::new()
            .with_pinned_device(DeviceLocation::Cpu)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn)
            .with_capabilities_for(&caps_fn)
            .with_transfer_estimator(&est)
            .with_input_residency(&residency_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");

        // All-CPU = 2400; migrate-at-n1 = 3×100 + 1000 (entry) +
        // 1000 (exit) = 2300 → the whole segment moves, THROUGH the
        // reshape.
        for id in [n1, n2, n3] {
            assert_eq!(
                plan.alternatives(id).unwrap().winner().unwrap().device,
                cuda0,
                "the chain must not break at the Reshape pass-through",
            );
        }
        assert!(plan.alternatives(r).is_none(), "Reshape carries no plan entry");
    }
}

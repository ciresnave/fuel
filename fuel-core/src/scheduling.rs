//! Orchestrator for the Phase 6b probe → judge → dispatch pipeline.
//!
//! Wraps the three pieces so callers get a ready-to-query
//! [`DispatchTable`] from one call. Does the right thing re: reuse
//! across restarts: if the persisted probe matches the current
//! hardware and a persisted profile is present, the Judge is
//! skipped entirely.
//!
//! # One-call API
//!
//! ```no_run
//! use fuel_core::scheduling::{prepare_dispatch_table, ScheduleOptions};
//! let (table, _report) = prepare_dispatch_table(ScheduleOptions::default())
//!     .expect("prepare_dispatch_table");
//! // `table` is now queryable with `.pick(op, dtype, size, criterion)`.
//! ```
//!
//! # Reuse rules
//!
//! 1. Probe the current hardware.
//! 2. If a persisted [`ProbeReport`] exists and `probe.diff(&prior)`
//!    is [`HardwareChange::Unchanged`], look for a persisted
//!    [`ProfileReport`].
//! 3. If a persisted profile is present **and** its schema version
//!    matches, **skip the Judge** and build the dispatch table from
//!    the persisted profile. Save cost on every startup.
//! 4. Otherwise re-run the Judge and persist both reports.
//!
//! # Where the reports live
//!
//! By default, both reports live under the OS cache dir
//! (`%LOCALAPPDATA%\fuel\` on Windows, `$XDG_CACHE_HOME/fuel/` on
//! Linux). Callers that want explicit paths (for CI, containers,
//! per-user config) can override via [`ScheduleOptions`].

use crate::judge::{Criterion, DispatchOptions, DispatchTable, Pick};
use crate::judge::{Judge, OpKind, ProfileEntry, ProfileReport, SizeClass};
use crate::probe::{HardwareChange, ProbeReport};
use crate::transfer_cost::BandwidthMatrix;
use fuel_core_types::probe::BackendId;
use fuel_core_types::{DType, DeviceLocation, Result};
use fuel_graph::{Graph, NodeId, Op};
use std::collections::HashMap;
use std::path::PathBuf;

/// Options for [`prepare_dispatch_table`] and the Phase 6c
/// [`prepare_dp_inputs`] / [`auto_place_and_route_with_transfer_cost`]
/// — lets callers override paths, force re-measurement, and tune
/// dispatch-table construction.
pub struct ScheduleOptions {
    /// Explicit path for the probe report. `None` = OS cache default.
    pub probe_path: Option<PathBuf>,
    /// Explicit path for the profile report. `None` = OS cache default.
    pub profile_path: Option<PathBuf>,
    /// Explicit path for the bandwidth report (Phase 6c). `None` =
    /// OS cache default (`bandwidth.json` next to probe.json).
    pub bandwidth_path: Option<PathBuf>,
    /// Force the Judge to re-run even if the persisted state is
    /// otherwise reusable. Useful after toolchain upgrades where a
    /// driver version bump didn't happen but compiler codegen
    /// changed.
    pub force_rejudge: bool,
    /// Force the bandwidth matrix to be re-measured. Same use case
    /// as `force_rejudge` for the transfer-cost half of Phase 6c.
    pub force_remeasure_bandwidth: bool,
    /// Judge config. Default = `Judge::default()`.
    pub judge: Judge,
    /// Dispatch table construction options.
    pub dispatch: DispatchOptions,
}

impl Default for ScheduleOptions {
    fn default() -> Self {
        Self {
            probe_path:                None,
            profile_path:              None,
            bandwidth_path:            None,
            force_rejudge:             false,
            force_remeasure_bandwidth: false,
            judge:                     Judge::default(),
            dispatch:                  DispatchOptions::default(),
        }
    }
}

/// Probe → [load / re-Judge] → build dispatch. Persists both reports
/// on a fresh Judge run. Returns the dispatch table paired with the
/// profile report it was built from (callers often want the raw
/// measurements too — for logging, debugging, or a custom secondary
/// dispatch table).
pub fn prepare_dispatch_table(
    opts: ScheduleOptions,
) -> Result<(DispatchTable, ProfileReport)> {
    let probe_path = opts.probe_path.clone()
        .or_else(crate::probe::default_report_path);
    let profile_path = opts.profile_path.clone()
        .or_else(crate::judge::default_report_path);

    let current_probe = ProbeReport::probe_all();

    // Step 1: decide if the persisted profile is reusable.
    let mut reuse_profile = false;
    if !opts.force_rejudge {
        if let Some(pp) = probe_path.as_ref() {
            if let Ok(Some(prior)) = ProbeReport::load(pp) {
                if matches!(current_probe.diff(&prior), HardwareChange::Unchanged) {
                    reuse_profile = true;
                }
            }
        }
    }

    let profile = if reuse_profile {
        if let Some(pp) = profile_path.as_ref() {
            match ProfileReport::load(pp)? {
                Some(r) => r,
                None => run_and_persist(&current_probe, &opts, &probe_path, &profile_path)?,
            }
        } else {
            run_and_persist(&current_probe, &opts, &probe_path, &profile_path)?
        }
    } else {
        run_and_persist(&current_probe, &opts, &probe_path, &profile_path)?
    };

    let table = DispatchTable::build_with(&profile, opts.dispatch);
    Ok((table, profile))
}

// ---- Placement recommender ----------------------------------------------
//
// Phase 6b's last machinery piece: walk a graph, ask the dispatch
// table where each node would run best, return a per-NodeId placement
// plan. The plan is the input that feeds into
// `fuel_graph::opt::insert_moves` (or future router-level placement
// passes); this function does NOT mutate the graph itself.
//
// Today the dispatch table only knows about MatMul + AddElementwise
// (the Judge's profile matrix). Every other op falls through to
// `fallback_device`. As the Judge's op coverage grows the
// recommender's coverage grows with it for free.

/// Convert a `fuel_graph::Op` to the Judge's `OpKind`. Returns `None`
/// for ops the Judge doesn't profile yet.
fn op_to_kind(op: &Op) -> Option<OpKind> {
    match op {
        Op::MatMul => Some(OpKind::MatMul),
        Op::Add    => Some(OpKind::AddElementwise),
        _          => None,
    }
}

/// Convert a dispatch-table [`Pick`] into the Fuel placement enum.
fn pick_to_location(pick: Pick) -> Option<DeviceLocation> {
    match pick.backend {
        BackendId::Cpu       => Some(DeviceLocation::Cpu),
        BackendId::Cuda      => Some(DeviceLocation::Cuda   { gpu_id: pick.device_index as usize }),
        BackendId::Vulkan    => Some(DeviceLocation::Vulkan { gpu_id: pick.device_index as usize }),
        BackendId::Metal     => Some(DeviceLocation::Metal  { gpu_id: pick.device_index as usize }),
        // BackendId is #[non_exhaustive]; future variants get the
        // CPU fallback until they have a placement rule of their own.
        _ => Some(DeviceLocation::Cpu),
    }
}

/// Walk every node in `graph` and produce a recommended
/// [`DeviceLocation`] per node. Ops the Judge profiled produce a
/// dispatch-table-driven recommendation; ops it didn't profile fall
/// back to `fallback_device` (typically the router's default).
///
/// The output is a `HashMap<NodeId, DeviceLocation>` ready to feed
/// into `fuel_graph::opt::insert_moves` or any other placement-
/// honouring pass.
///
/// # When to use
///
/// The recommender is intended for **realize-time** placement
/// decisions, after the orchestrator has produced a [`DispatchTable`].
/// It does not mutate the graph; it just produces a plan. Callers
/// can override individual nodes (pin a specific tensor to CPU for
/// debugging, force a backbone onto a GPU regardless of size, etc.)
/// before passing the plan to `insert_moves`.
///
/// # Tolerance for unprofiled ops
///
/// Most ops in any real graph won't be profiled in v1 (Judge today
/// covers MatMul + AddElementwise). Those nodes get
/// `fallback_device`, which means the dispatch table is acting as a
/// "preferred placement for hot ops" overlay rather than a hard
/// scheduling decision. As the Judge's op coverage grows the overlay
/// grows.
pub fn recommend_placement(
    graph: &Graph,
    table: &DispatchTable,
    criterion: Criterion,
    fallback_device: DeviceLocation,
) -> HashMap<NodeId, DeviceLocation> {
    let mut plan = HashMap::with_capacity(graph.len());
    for i in 0..graph.len() {
        let id = NodeId(i);
        let node = graph.node(id);
        let device = recommend_for_node(node, table, criterion).unwrap_or(fallback_device);
        plan.insert(id, device);
    }
    plan
}

/// Single-node recommendation. Public so callers that already have
/// a node in hand (e.g. during a custom graph rewrite pass) can ask
/// the dispatch table directly without building the full plan.
pub fn recommend_for_node(
    node: &fuel_graph::Node,
    table: &DispatchTable,
    criterion: Criterion,
) -> Option<DeviceLocation> {
    let kind = op_to_kind(&node.op)?;
    if node.dtype != DType::F32 {
        // Judge only covers F32 today. Other dtypes get fallback.
        return None;
    }
    let size_class = SizeClass::from_elem_count(node.shape.elem_count());
    let pick = table.pick_nearest(kind, node.dtype, size_class, criterion)?;
    pick_to_location(pick)
}

/// Apply a placement plan to a [`Graph`] — set per-node placement
/// hints for every entry **that doesn't already have one**. Skipping
/// pre-set hints is the safe default: a user who explicitly pinned a
/// node to a device shouldn't have that overridden by the dispatch
/// table's recommendation. Use [`force_apply_placement_plan`] when
/// the override is intentional.
///
/// After this, a follow-up [`fuel_graph::opt::insert_copies`] pass
/// will see the per-node hints and insert `Op::Copy` nodes at the
/// boundaries between devices that disagree.
pub fn apply_placement_plan(
    graph: &fuel_graph::SharedGraph,
    plan: &HashMap<NodeId, DeviceLocation>,
) {
    let mut g = graph.write().unwrap();
    for (&id, &loc) in plan {
        if g.placement(id).is_none() {
            g.set_placement(id, loc);
        }
    }
}

/// Like [`apply_placement_plan`] but overwrites existing hints.
/// Useful when a re-Judge produces fresher recommendations or when
/// a user wants to dispatch-route a graph that's been hand-pinned
/// in an earlier pass.
pub fn force_apply_placement_plan(
    graph: &fuel_graph::SharedGraph,
    plan: &HashMap<NodeId, DeviceLocation>,
) {
    let mut g = graph.write().unwrap();
    for (&id, &loc) in plan {
        g.set_placement(id, loc);
    }
}

/// One-call Phase 6b auto-router: runs [`recommend_placement`],
/// applies the resulting hints (skip-existing semantics), and runs
/// [`fuel_graph::opt::insert_copies`] to materialise the cross-
/// device transfers. Returns the new root IDs from `insert_copies`
/// (which may have been remapped if any of `roots`'s ops were
/// rewritten as part of inserting Copy nodes).
///
/// This is the "do the right thing" entry point for users who want
/// the Phase 6b dispatch table to drive placement automatically.
/// Users who want finer control compose the three steps manually.
pub fn auto_place_and_route(
    graph: &fuel_graph::SharedGraph,
    roots: &[NodeId],
    table: &DispatchTable,
    criterion: Criterion,
    fallback_device: DeviceLocation,
) -> Vec<NodeId> {
    let plan = {
        let g = graph.read().unwrap();
        recommend_placement(&g, table, criterion, fallback_device)
    };
    apply_placement_plan(graph, &plan);
    fuel_graph::opt::insert_copies(graph, roots)
}

/// Default cache path for the bandwidth report — `bandwidth.json`
/// next to `probe.json` in the OS cache dir.
fn default_bandwidth_path() -> Option<PathBuf> {
    crate::probe::default_report_path()
        .and_then(|p| p.parent().map(|parent| parent.join(crate::transfer_cost::BANDWIDTH_REPORT_FILENAME)))
}

/// Probe → load-or-judge → load-or-measure-bandwidth, returning all
/// three Phase 6c inputs in one call. Reuses persisted reports when
/// the current probe matches the prior one (same hardware-change
/// rule the Judge uses); re-measures any piece marked
/// `force_*` in the options or that's missing a persisted file.
///
/// Both reports persist to disk on a fresh measurement so subsequent
/// calls warm-start.
pub fn prepare_dp_inputs(
    opts: ScheduleOptions,
) -> Result<(ProbeReport, ProfileReport, BandwidthMatrix)> {
    let probe_path     = opts.probe_path.clone().or_else(crate::probe::default_report_path);
    let profile_path   = opts.profile_path.clone().or_else(crate::judge::default_report_path);
    let bandwidth_path = opts.bandwidth_path.clone().or_else(default_bandwidth_path);

    let current_probe = ProbeReport::probe_all();

    // Decide reuse eligibility once — same rule for both judge and
    // bandwidth caches. Hardware-change diff is the source of truth.
    let mut hardware_unchanged = false;
    if let Some(pp) = probe_path.as_ref() {
        if let Ok(Some(prior)) = ProbeReport::load(pp) {
            if matches!(current_probe.diff(&prior), HardwareChange::Unchanged) {
                hardware_unchanged = true;
            }
        }
    }

    // -- Profile report --
    let profile = if hardware_unchanged && !opts.force_rejudge {
        if let Some(pp) = profile_path.as_ref() {
            ProfileReport::load(pp)?.unwrap_or_else(|| opts.judge.run(&current_probe))
        } else {
            opts.judge.run(&current_probe)
        }
    } else {
        opts.judge.run(&current_probe)
    };

    // -- Bandwidth matrix --
    let bandwidth = if hardware_unchanged && !opts.force_remeasure_bandwidth {
        if let Some(bp) = bandwidth_path.as_ref() {
            BandwidthMatrix::load(bp)?.unwrap_or_else(|| BandwidthMatrix::measure(&current_probe))
        } else {
            BandwidthMatrix::measure(&current_probe)
        }
    } else {
        BandwidthMatrix::measure(&current_probe)
    };

    // -- Persist (best-effort) --
    if let Some(p) = probe_path.as_ref() {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = current_probe.save(p) {
            eprintln!("fuel scheduling: failed to persist probe report to {p:?}: {e}");
        }
    }
    if let Some(p) = profile_path.as_ref() {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = profile.save(p) {
            eprintln!("fuel scheduling: failed to persist profile report to {p:?}: {e}");
        }
    }
    if let Some(p) = bandwidth_path.as_ref() {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = bandwidth.save(p) {
            eprintln!("fuel scheduling: failed to persist bandwidth report to {p:?}: {e}");
        }
    }

    Ok((current_probe, profile, bandwidth))
}

/// One-call Phase 6c auto-router: probe → load-or-judge → load-or-
/// measure-bandwidth → DP-plan → apply hints → insert_copies. Returns
/// the new root IDs after `Op::Copy` insertion.
///
/// This is the transfer-cost-aware analogue of the Phase 6b
/// [`auto_place_and_route`]. The DP planner accounts for both
/// per-op compute cost (from the Judge) and per-edge transfer cost
/// (from the bandwidth matrix), producing placements that minimise
/// the total cost rather than just the sum of per-node winners.
///
/// Unprofiled-op nodes (Const, GELU, etc.) take `fallback_device`.
pub fn auto_place_and_route_with_transfer_cost(
    graph: &fuel_graph::SharedGraph,
    roots: &[NodeId],
    opts: ScheduleOptions,
    fallback_device: DeviceLocation,
) -> Result<Vec<NodeId>> {
    let (probe, profile, bandwidth) = prepare_dp_inputs(opts)?;
    let mut backends_seen: std::collections::HashSet<BackendId> = Default::default();
    let mut available_backends: Vec<BackendId> = Vec::new();
    for d in &probe.devices {
        if backends_seen.insert(d.backend) {
            available_backends.push(d.backend);
        }
    }
    let plan = {
        let g = graph.read().unwrap();
        dp_plan(&g, roots, &profile, &bandwidth, &available_backends, fallback_device)
    };
    apply_placement_plan(graph, &plan);
    Ok(fuel_graph::opt::insert_copies(graph, roots))
}

// ---- Phase 6c: transfer-cost-aware DP planner ----------------------------

/// Phase 6c DP-based placement planner. Produces a per-node placement
/// that minimises **total cost = compute + transfer**, via forward
/// dynamic programming over the topo-sorted graph.
///
/// For each node in topo order and for each candidate backend `b`,
/// we compute:
///
/// ```text
///   best_cost[node, b] = compute_cost(node, b) +
///                        Σ_inputs min_{b_i} ( best_cost[input, b_i]
///                                          + transfer_cost(b_i → b, input_bytes) )
/// ```
///
/// After the forward pass, each root picks its min-cost backend, and
/// a backtrack pass propagates the chosen backend to every reachable
/// input. Nodes that aren't profiled (op kind not in the dispatch
/// table, e.g. ConvNeXt's GELU today) get the `fallback_device` and
/// don't contribute meaningful compute cost — but their inputs still
/// pay transfer costs to wherever they end up.
///
/// Compared to [`recommend_placement`], the DP planner accounts for
/// the cost of the inputs being on a different backend. That changes
/// the right answer in cases where a fast backend's transfer cost
/// exceeds the compute speedup.
pub fn dp_plan(
    graph: &fuel_graph::Graph,
    roots: &[NodeId],
    profile: &ProfileReport,
    bandwidth: &crate::transfer_cost::BandwidthMatrix,
    available_backends: &[BackendId],
    fallback_device: DeviceLocation,
) -> HashMap<NodeId, DeviceLocation> {
    use fuel_graph::topo_order_multi;
    let order = topo_order_multi(graph, roots);

    // (node, backend) → (cumulative_cost_ns, [chosen_backend_per_input]).
    let mut cost: HashMap<(NodeId, BackendId), (f64, Vec<BackendId>)> = HashMap::new();

    let backends = if available_backends.is_empty() {
        // Default: fall back to CPU only; gives sensible behaviour when
        // the caller forgot to enumerate. Equivalent to recommend_placement
        // with no GPU options.
        vec![BackendId::Cpu]
    } else {
        available_backends.to_vec()
    };

    for &id in &order {
        let node = graph.node(id);
        let kind = op_to_kind(&node.op);
        let size_class = SizeClass::from_elem_count(node.shape.elem_count());
        let dtype = node.dtype;
        // Const nodes live in host memory; consumers on non-CPU
        // backends pay the upload. Pin them to CPU so the DP can't
        // "place" a Const on CUDA for free.
        let is_const = matches!(node.op, fuel_graph::Op::Const);

        for &b in &backends {
            // Const ops only have a finite cost on CPU. Non-CPU
            // backends see Const-on-{CUDA, Vulkan, ...} as infinite,
            // forcing consumers to transfer.
            if is_const && b != BackendId::Cpu {
                cost.insert((id, b), (f64::INFINITY, vec![]));
                continue;
            }
            // Compute cost on backend b for this op. Read directly
            // from the profile report — we want the actual measured
            // latency for THIS (op, dtype, size, backend), not just
            // "did this backend win the dispatch table." If the
            // backend isn't in the profile (didn't measure it),
            // fall back to a generic penalty so the DP can still
            // make progress.
            let compute = if let Some(k) = kind {
                if dtype == DType::F32 {
                    profile_lookup_latency_ns(profile, k, dtype, size_class, b)
                        .unwrap_or(UNPROFILED_BACKEND_PENALTY_NS)
                } else {
                    UNPROFILED_BACKEND_PENALTY_NS
                }
            } else {
                // Op not profiled (e.g. GELU, Mul). Assign a small
                // backend-agnostic compute estimate so transfer
                // costs dominate routing.
                UNPROFILED_OP_COMPUTE_NS
            };

            // For each input, pick min over b_i of [input_cost(b_i) + transfer(b_i, b)].
            let mut total_cost = compute;
            let mut input_choices = Vec::with_capacity(node.inputs.len());
            for &input in &node.inputs {
                let input_node = graph.node(input);
                let bytes = input_node.shape.elem_count() * dtype_size_bytes(input_node.dtype);
                let mut best = f64::INFINITY;
                let mut best_b = backends[0];
                for &b_i in &backends {
                    let prior = cost.get(&(input, b_i)).map(|(c, _)| *c).unwrap_or(f64::INFINITY);
                    if !prior.is_finite() { continue; }
                    let xfer = transfer_cost_with_cpu_fallback(bandwidth, b_i, b, bytes);
                    let candidate = prior + xfer;
                    if candidate < best {
                        best   = candidate;
                        best_b = b_i;
                    }
                }
                if best.is_finite() {
                    total_cost += best;
                    input_choices.push(best_b);
                } else {
                    // No reachable input cost — this (node, b) is
                    // unreachable. Mark with infinity so it loses
                    // backtracking.
                    total_cost = f64::INFINITY;
                    break;
                }
            }
            cost.insert((id, b), (total_cost, input_choices));
        }
    }

    // Pick min-cost backend for each root, then backtrack.
    let mut placement: HashMap<NodeId, BackendId> = HashMap::new();
    for &root in roots {
        let mut best = f64::INFINITY;
        let mut best_b = backends[0];
        for &b in &backends {
            if let Some(&(c, _)) = cost.get(&(root, b)).map(|x| x).iter().next().copied() {
                if c < best {
                    best   = c;
                    best_b = b;
                }
            }
        }
        placement.insert(root, best_b);
    }

    // Backtrack: for each placed node, propagate input choices.
    let mut stack: Vec<NodeId> = roots.to_vec();
    while let Some(id) = stack.pop() {
        let chosen_b = match placement.get(&id) {
            Some(&b) => b,
            None     => continue,
        };
        if let Some((_, inputs_chosen)) = cost.get(&(id, chosen_b)).cloned() {
            let node = graph.node(id);
            for (idx, &input) in node.inputs.iter().enumerate() {
                if let Some(&b_i) = inputs_chosen.get(idx) {
                    if !placement.contains_key(&input) {
                        placement.insert(input, b_i);
                        stack.push(input);
                    }
                }
            }
        }
    }

    // Translate to DeviceLocation. Backends not in the dispatch's
    // pick_to_location table fall back to `fallback_device`.
    placement
        .into_iter()
        .map(|(id, b)| {
            let pick = crate::judge::Pick { backend: b, device_index: 0 };
            (id, pick_to_location(pick).unwrap_or(fallback_device))
        })
        .collect()
}

/// Per-byte transfer cost from `src` to `dst`. If no direct entry,
/// route via CPU as a two-hop transfer (the worst case is the
/// Vulkan→CPU→CUDA pattern; CPU→CPU is essentially free).
fn transfer_cost_with_cpu_fallback(
    bandwidth: &crate::transfer_cost::BandwidthMatrix,
    src: BackendId,
    dst: BackendId,
    bytes: usize,
) -> f64 {
    if src == dst {
        return 0.0;
    }
    if let Some(t) = bandwidth.lookup(src, dst) {
        return t.ns_per_byte * bytes as f64;
    }
    // Two-hop via CPU.
    let leg1 = bandwidth.lookup(src, BackendId::Cpu).map(|t| t.ns_per_byte);
    let leg2 = bandwidth.lookup(BackendId::Cpu, dst).map(|t| t.ns_per_byte);
    match (leg1, leg2) {
        (Some(a), Some(b)) => (a + b) * bytes as f64,
        _ => f64::INFINITY,  // unreachable pair
    }
}

fn dtype_size_bytes(d: DType) -> usize {
    match d {
        DType::F32 | DType::U32 | DType::I32 => 4,
        DType::F64 | DType::I64 => 8,
        DType::F16 | DType::BF16 | DType::I16 => 2,
        DType::U8 | DType::I8 | DType::F8E4M3 | DType::F8E8M0 => 1,
        DType::F4 => 1,         // packed; per-element fractional, round up
        DType::F6E2M3 | DType::F6E3M2 => 1,
    }
}

/// Find the closest measured `(op, dtype, size_class, backend)`
/// latency in the profile report. Returns `None` if no entry for
/// the requested backend exists at any size class for this
/// (op, dtype). Walks the profile linearly — fine for v1; a
/// pre-built index would be cheap to add later if profiles get
/// large.
fn profile_lookup_latency_ns(
    profile: &ProfileReport,
    op: OpKind,
    dtype: DType,
    size_class: SizeClass,
    backend: BackendId,
) -> Option<f64> {
    // Exact size-class match first.
    if let Some(e) = profile.entries.iter().find(|e|
        e.op == op && e.dtype == dtype && e.size_class == size_class && e.backend == backend
    ) {
        return Some(e.latency_ns as f64);
    }
    // Nearest-size fallback (same op/dtype/backend, closest size).
    let candidates: Vec<&ProfileEntry> = profile.entries.iter()
        .filter(|e| e.op == op && e.dtype == dtype && e.backend == backend)
        .collect();
    if candidates.is_empty() { return None; }
    let target = size_class.0 as i32;
    let nearest = candidates.iter()
        .min_by_key(|e| (e.size_class.0 as i32 - target).abs())
        .unwrap();
    Some(nearest.latency_ns as f64)
}

/// Penalty (in ns) for a backend that has no profile entry for the
/// op at any size. Should be high enough that the DP avoids picking
/// it when other backends DO have profiles, but not infinite — a
/// genuine "no choice" case (every backend missing) should still
/// produce a placement rather than panicking.
const UNPROFILED_BACKEND_PENALTY_NS: f64 = 1_000_000_000.0;  // 1 second

/// Compute cost for ops whose op kind isn't profiled. Should be
/// small relative to transfer costs so unprofiled ops route along
/// with their inputs by default.
const UNPROFILED_OP_COMPUTE_NS: f64 = 1_000.0;

fn run_and_persist(
    probe: &ProbeReport,
    opts: &ScheduleOptions,
    probe_path: &Option<PathBuf>,
    profile_path: &Option<PathBuf>,
) -> Result<ProfileReport> {
    let profile = opts.judge.run(probe);

    // Best-effort persistence: if the parent dir doesn't exist or a
    // write fails, log to stderr and continue. Dispatch decisions
    // still work in-memory even when we can't write the reports.
    if let Some(p) = probe_path {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = probe.save(p) {
            eprintln!("fuel scheduling: failed to persist probe report to {p:?}: {e}");
        }
    }
    if let Some(p) = profile_path {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = profile.save(p) {
            eprintln!("fuel scheduling: failed to persist profile report to {p:?}: {e}");
        }
    }

    Ok(profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::{OpKind, OpSize};

    /// Force-rejudge on a fresh scratch dir — verifies the full
    /// orchestrator path end-to-end (no prior state, run everything,
    /// persist, reload).
    #[test]
    fn end_to_end_probe_judge_dispatch() {
        let scratch = std::env::temp_dir().join(format!(
            "fuel-schedule-test-{}", std::process::id()
        ));
        let _ = std::fs::create_dir_all(&scratch);

        let opts = ScheduleOptions {
            probe_path:                Some(scratch.join("probe.json")),
            profile_path:              Some(scratch.join("judge.json")),
            bandwidth_path:            Some(scratch.join("bandwidth.json")),
            force_rejudge:             true,
            force_remeasure_bandwidth: true,
            judge: Judge {
                iterations: 3,
                warmup: 1,
                size_plan_override: Some(vec![
                    (OpKind::MatMul, OpSize::MatMul { m: 32, n: 32, k: 32 }),
                ]),
            },
            dispatch: Default::default(),
        };

        let (table, profile) = prepare_dispatch_table(opts).expect("prepare");
        assert!(profile.entries.len() >= 2, "profile should have cpu + ref entries");
        assert!(table.len() >= 1, "dispatch table should have at least one entry");

        // Both files should exist now.
        assert!(scratch.join("probe.json").exists());
        assert!(scratch.join("judge.json").exists());

        let _ = std::fs::remove_dir_all(&scratch);
    }

    /// Second call with matching hardware and no force_rejudge should
    /// skip the Judge (observable as "profile entries come from the
    /// persisted file, not a fresh measurement"). We can't assert
    /// exact byte equality because the device's description timing
    /// varies, but entry count should be stable.
    #[test]
    fn reuses_profile_when_hardware_unchanged() {
        let scratch = std::env::temp_dir().join(format!(
            "fuel-schedule-reuse-{}", std::process::id()
        ));
        let _ = std::fs::create_dir_all(&scratch);

        let tiny_judge = || Judge {
            iterations: 3, warmup: 1,
            size_plan_override: Some(vec![
                (OpKind::MatMul, OpSize::MatMul { m: 16, n: 16, k: 16 }),
            ]),
        };

        let first = prepare_dispatch_table(ScheduleOptions {
            probe_path:                Some(scratch.join("probe.json")),
            profile_path:              Some(scratch.join("judge.json")),
            bandwidth_path:            Some(scratch.join("bandwidth.json")),
            force_rejudge:             true,
            force_remeasure_bandwidth: true,
            judge:                     tiny_judge(),
            dispatch:                  Default::default(),
        }).expect("first run");

        let second = prepare_dispatch_table(ScheduleOptions {
            probe_path:                Some(scratch.join("probe.json")),
            profile_path:              Some(scratch.join("judge.json")),
            bandwidth_path:            Some(scratch.join("bandwidth.json")),
            force_rejudge:             false,  // reuse eligible
            force_remeasure_bandwidth: false,
            judge:                     tiny_judge(),
            dispatch:                  Default::default(),
        }).expect("second run");

        // Second run's profile should equal first's (loaded from disk,
        // not re-measured). Latency fields would differ on re-measure
        // — identical if loaded from disk.
        assert_eq!(first.1, second.1,
            "hardware unchanged + no force_rejudge should yield identical profile");

        let _ = std::fs::remove_dir_all(&scratch);
    }

    use crate::judge::Criterion;
    use crate::judge::{ProfileEntry, ProfileReport, PROFILE_REPORT_VERSION};
    use fuel_graph::Tensor;
    use fuel_core_types::Shape;
    use std::sync::Arc;

    /// Hand-build a tiny dispatch table where CUDA wins MatMul at
    /// large sizes and CPU wins at small. Then build a graph with
    /// matmuls of varying sizes plus an unprofiled op (Mul) and
    /// verify the recommender sends each node where the table
    /// predicts.
    #[test]
    fn recommend_placement_routes_per_node() {
        // -- 1) hand-craft a profile report with deliberate winners --
        let mk = |backend: BackendId, size: u8, latency: u64| ProfileEntry {
            op: OpKind::MatMul,
            dtype: DType::F32,
            size_class: SizeClass(size),
            backend,
            device_index: 0,
            latency_ns: latency,
            iterations: 1,
            max_rel_error: 0.0,
        };
        let report = ProfileReport {
            version: PROFILE_REPORT_VERSION,
            entries: vec![
                // size 12 (≈ 64×64 matmul): CPU wins (10μs vs 2ms CUDA launch overhead)
                mk(BackendId::Cpu,  12,    10_000),
                mk(BackendId::Cuda, 12, 2_000_000),
                // size 20 (≈ 1024×1024 matmul): CUDA wins (5ms vs 50ms CPU)
                mk(BackendId::Cpu,  20, 50_000_000),
                mk(BackendId::Cuda, 20,  5_000_000),
            ],
        };
        let table = DispatchTable::build(&report);

        // -- 2) build a graph with two matmuls + one unprofiled op
        //       — all derived from a single root so they share a graph.
        let small_a = Tensor::from_f32(vec![0.0_f32; 64 * 64], Shape::from_dims(&[64, 64]), crate::Device::cpu().as_dyn());
        let small_b = small_a.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; 64 * 64]),
            Shape::from_dims(&[64, 64]),
        );
        let small_mm = small_a.matmul(&small_b);  // size_class = 12 (4096 elements)

        // Build the big matmul as constants on the SAME graph as small_a.
        let big_a = small_a.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; 1024 * 1024]),
            Shape::from_dims(&[1024, 1024]),
        );
        let big_b = small_a.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; 1024 * 1024]),
            Shape::from_dims(&[1024, 1024]),
        );
        let big_mm = big_a.matmul(&big_b);  // size_class = 20

        // Unprofiled op (Sub) on small tensors — should fall back.
        let unprofiled = small_mm.sub(&small_b);

        // The graph for the recommender is whichever one any of our
        // tensors point into; they all share the same graph since
        // they were built from the same `from_f32` root.
        let graph = small_a.graph().read().unwrap();

        let plan = recommend_placement(
            &graph,
            &table,
            Criterion::Fastest,
            DeviceLocation::Cpu,
        );

        // small matmul → CPU (size_class 12 winner)
        assert_eq!(plan[&small_mm.id()], DeviceLocation::Cpu);
        // big matmul → CUDA (size_class 20 winner)
        assert_eq!(plan[&big_mm.id()], DeviceLocation::Cuda { gpu_id: 0 });
        // Unprofiled sub-op → fallback (CPU)
        assert_eq!(plan[&unprofiled.id()], DeviceLocation::Cpu);
        // Const nodes → fallback (no Op::Const → OpKind mapping)
        assert!(matches!(plan[&small_a.id()], DeviceLocation::Cpu));
    }

    /// Skip-existing semantics: if a node already has a placement
    /// hint, the auto-router must not overwrite it. The user's
    /// explicit `set_placement` call is authoritative.
    #[test]
    fn apply_placement_plan_skips_pre_set_hints() {
        let entries = vec![
            ProfileEntry { op: OpKind::MatMul, dtype: DType::F32, size_class: SizeClass(20),
                backend: BackendId::Cpu,  device_index: 0, latency_ns: 50_000_000, iterations: 1, max_rel_error: 0.0 },
            ProfileEntry { op: OpKind::MatMul, dtype: DType::F32, size_class: SizeClass(20),
                backend: BackendId::Cuda, device_index: 0, latency_ns:  5_000_000, iterations: 1, max_rel_error: 0.0 },
        ];
        let report = ProfileReport { version: PROFILE_REPORT_VERSION, entries };
        let table = DispatchTable::build(&report);

        let a = Tensor::from_f32(vec![0.0_f32; 1024 * 1024], Shape::from_dims(&[1024, 1024]), crate::Device::cpu().as_dyn());
        let b = a.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; 1024 * 1024]),
            Shape::from_dims(&[1024, 1024]),
        );
        let mm = a.matmul(&b);  // dispatch table would pick CUDA

        // User pins it to CPU explicitly.
        a.graph().write().unwrap().set_placement(mm.id(), DeviceLocation::Cpu);

        let plan = recommend_placement(
            &a.graph().read().unwrap(),
            &table,
            Criterion::Fastest,
            DeviceLocation::Cpu,
        );
        // Recommendation says CUDA, but apply_placement_plan must respect the existing CPU hint.
        assert_eq!(plan[&mm.id()], DeviceLocation::Cuda { gpu_id: 0 });
        apply_placement_plan(a.graph(), &plan);
        assert_eq!(
            a.graph().read().unwrap().placement(mm.id()),
            Some(DeviceLocation::Cpu),
            "user's explicit hint must not be overwritten by recommend_placement",
        );
    }

    /// Phase 6c orchestrator end-to-end: probe → judge → bandwidth →
    /// dp_plan → apply → insert_copies. Verifies the full chain
    /// completes without panicking on the dev rig and that the
    /// persisted reports are written.
    #[test]
    fn end_to_end_dp_orchestrator() {
        let scratch = std::env::temp_dir().join(format!(
            "fuel-dp-orch-test-{}", std::process::id()
        ));
        let _ = std::fs::create_dir_all(&scratch);

        // Build a small graph the planner will actually route.
        let a = Tensor::from_f32(vec![0.0_f32; 64 * 64], Shape::from_dims(&[64, 64]), crate::Device::cpu().as_dyn());
        let b = a.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; 64 * 64]),
            Shape::from_dims(&[64, 64]),
        );
        let mm = a.matmul(&b);

        let opts = ScheduleOptions {
            probe_path:                Some(scratch.join("probe.json")),
            profile_path:              Some(scratch.join("judge.json")),
            bandwidth_path:            Some(scratch.join("bandwidth.json")),
            force_rejudge:             true,
            force_remeasure_bandwidth: true,
            judge: Judge {
                iterations: 3,
                warmup: 1,
                size_plan_override: Some(vec![
                    (OpKind::MatMul, OpSize::MatMul { m: 32, n: 32, k: 32 }),
                ]),
            },
            dispatch: Default::default(),
        };

        let new_roots = auto_place_and_route_with_transfer_cost(
            a.graph(),
            &[mm.id()],
            opts,
            DeviceLocation::Cpu,
        ).expect("orchestrator");

        // We don't assert specific placement (depends on rig) — just
        // that the orchestrator runs without panic and emits roots.
        assert!(!new_roots.is_empty());
        assert!(scratch.join("probe.json").exists());
        assert!(scratch.join("judge.json").exists());
        assert!(scratch.join("bandwidth.json").exists());

        let _ = std::fs::remove_dir_all(&scratch);
    }

    /// Phase 6c: a transfer-dominated single-op graph routes to CPU
    /// even though the dispatch table picks CUDA — because the input
    /// is on CPU and the H2D + D2H round-trip costs more than the
    /// compute saving. Without transfer cost the planner would pick
    /// CUDA; with it, CPU wins.
    #[test]
    fn dp_plan_avoids_costly_transfers() {
        use crate::transfer_cost::{BandwidthMatrix, TransferCost, BANDWIDTH_REPORT_VERSION};
        // CUDA is the winner of dispatch (small compute cost penalty)
        // but every byte costs 100ns to upload + 100ns to download.
        // Even a tiny tensor crosses the threshold where keeping it
        // on CPU is cheaper.
        let entries = vec![
            ProfileEntry { op: OpKind::MatMul, dtype: DType::F32, size_class: SizeClass(12),
                backend: BackendId::Cpu,  device_index: 0, latency_ns:    10_000, iterations: 1, max_rel_error: 0.0 },
            ProfileEntry { op: OpKind::MatMul, dtype: DType::F32, size_class: SizeClass(12),
                backend: BackendId::Cuda, device_index: 0, latency_ns:     5_000, iterations: 1, max_rel_error: 0.0 },
        ];
        let report = ProfileReport { version: PROFILE_REPORT_VERSION, entries };
        let table = DispatchTable::build(&report);

        // Punitive bandwidth: 100 ns/byte each way. A 64×64 f32
        // matmul output is 64*64*4 = 16384 bytes = 1.6ms one-way
        // transfer = 3.2ms round-trip. The compute saving on CUDA
        // is 5µs vs 10µs CPU — only 5µs. Transfer cost dwarfs it.
        let bandwidth = BandwidthMatrix {
            version: BANDWIDTH_REPORT_VERSION,
            measurement_bytes: 1 << 24,
            entries: vec![
                TransferCost { src: BackendId::Cpu,  dst: BackendId::Cpu,  ns_per_byte:   0.05 },
                TransferCost { src: BackendId::Cpu,  dst: BackendId::Cuda, ns_per_byte: 100.0  },
                TransferCost { src: BackendId::Cuda, dst: BackendId::Cpu,  ns_per_byte: 100.0  },
            ],
        };

        let small_a = Tensor::from_f32(vec![0.0_f32; 64 * 64], Shape::from_dims(&[64, 64]), crate::Device::cpu().as_dyn());
        let small_b = small_a.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; 64 * 64]),
            Shape::from_dims(&[64, 64]),
        );
        let mm = small_a.matmul(&small_b);

        let plan = dp_plan(
            &small_a.graph().read().unwrap(),
            &[mm.id()],
            &report,
            &bandwidth,
            &[BackendId::Cpu, BackendId::Cuda],
            DeviceLocation::Cpu,
        );

        // The dispatch table's first-choice would have been CUDA
        // (`recommend_placement` produces CUDA here too). The DP
        // planner accounts for the transfer cost and picks CPU.
        assert_eq!(plan[&mm.id()], DeviceLocation::Cpu,
            "DP planner should pick CPU when transfer cost dominates");
    }

    /// Phase 6c: when transfer is cheap enough, the DP planner DOES
    /// route to the dispatch winner. Sanity check that the planner
    /// isn't pessimistically pinning everything to CPU.
    #[test]
    fn dp_plan_picks_dispatch_winner_when_transfer_is_cheap() {
        use crate::transfer_cost::{BandwidthMatrix, TransferCost, BANDWIDTH_REPORT_VERSION};
        let entries = vec![
            ProfileEntry { op: OpKind::MatMul, dtype: DType::F32, size_class: SizeClass(20),
                backend: BackendId::Cpu,  device_index: 0, latency_ns: 50_000_000, iterations: 1, max_rel_error: 0.0 },
            ProfileEntry { op: OpKind::MatMul, dtype: DType::F32, size_class: SizeClass(20),
                backend: BackendId::Cuda, device_index: 0, latency_ns:  5_000_000, iterations: 1, max_rel_error: 0.0 },
        ];
        let report = ProfileReport { version: PROFILE_REPORT_VERSION, entries };
        let table = DispatchTable::build(&report);
        // Realistic-ish PCIe bandwidth: 0.15 ns/byte (≈6.5 GB/s).
        let bandwidth = BandwidthMatrix {
            version: BANDWIDTH_REPORT_VERSION,
            measurement_bytes: 1 << 24,
            entries: vec![
                TransferCost { src: BackendId::Cpu,  dst: BackendId::Cpu,  ns_per_byte: 0.05 },
                TransferCost { src: BackendId::Cpu,  dst: BackendId::Cuda, ns_per_byte: 0.15 },
                TransferCost { src: BackendId::Cuda, dst: BackendId::Cpu,  ns_per_byte: 0.20 },
            ],
        };

        let big_a = Tensor::from_f32(vec![0.0_f32; 1024 * 1024], Shape::from_dims(&[1024, 1024]), crate::Device::cpu().as_dyn());
        let big_b = big_a.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; 1024 * 1024]),
            Shape::from_dims(&[1024, 1024]),
        );
        let mm = big_a.matmul(&big_b);

        let plan = dp_plan(
            &big_a.graph().read().unwrap(),
            &[mm.id()],
            &report,
            &bandwidth,
            &[BackendId::Cpu, BackendId::Cuda],
            DeviceLocation::Cpu,
        );

        // 1024² f32 = 4 MiB transfer ≈ 0.6ms each way = 1.2ms round-trip;
        // CUDA saves 45ms vs CPU. CUDA wins easily.
        assert_eq!(plan[&mm.id()], DeviceLocation::Cuda { gpu_id: 0 });
    }

    /// `auto_place_and_route` does the full dance in one call:
    /// recommend → apply (skip-existing) → insert_copies.
    #[test]
    fn auto_place_and_route_inserts_copies_for_split_graph() {
        let entries = vec![
            ProfileEntry { op: OpKind::MatMul, dtype: DType::F32, size_class: SizeClass(12),
                backend: BackendId::Cpu,  device_index: 0, latency_ns:    10_000, iterations: 1, max_rel_error: 0.0 },
            ProfileEntry { op: OpKind::MatMul, dtype: DType::F32, size_class: SizeClass(12),
                backend: BackendId::Cuda, device_index: 0, latency_ns: 2_000_000, iterations: 1, max_rel_error: 0.0 },
            ProfileEntry { op: OpKind::MatMul, dtype: DType::F32, size_class: SizeClass(20),
                backend: BackendId::Cpu,  device_index: 0, latency_ns: 50_000_000, iterations: 1, max_rel_error: 0.0 },
            ProfileEntry { op: OpKind::MatMul, dtype: DType::F32, size_class: SizeClass(20),
                backend: BackendId::Cuda, device_index: 0, latency_ns:  5_000_000, iterations: 1, max_rel_error: 0.0 },
        ];
        let report = ProfileReport { version: PROFILE_REPORT_VERSION, entries };
        let table = DispatchTable::build(&report);

        let small_a = Tensor::from_f32(vec![0.0_f32; 64 * 64], Shape::from_dims(&[64, 64]), crate::Device::cpu().as_dyn());
        let small_b = small_a.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; 64 * 64]),
            Shape::from_dims(&[64, 64]),
        );
        let small_mm = small_a.matmul(&small_b);  // → CPU

        let big_a = small_a.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; 1024 * 1024]),
            Shape::from_dims(&[1024, 1024]),
        );
        let big_b = small_a.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; 1024 * 1024]),
            Shape::from_dims(&[1024, 1024]),
        );
        let big_mm = big_a.matmul(&big_b);  // → CUDA

        let n_before = small_a.graph().read().unwrap().len();
        let _new_roots = auto_place_and_route(
            small_a.graph(),
            &[small_mm.id(), big_mm.id()],
            &table,
            Criterion::Fastest,
            DeviceLocation::Cpu,
        );
        let n_after = small_a.graph().read().unwrap().len();

        // Same expectation as the manual-pipeline test: ≥2 Copy nodes
        // targeting CUDA appeared at the boundary between the
        // CPU-placed sub-graph and the CUDA-placed sub-graph.
        let g = small_a.graph().read().unwrap();
        let mut cuda_copies = 0;
        for i in n_before..n_after {
            if let fuel_graph::Op::Copy { target } = &g.node(NodeId(i)).op {
                if matches!(target, DeviceLocation::Cuda { .. }) {
                    cuda_copies += 1;
                }
            }
        }
        assert!(
            cuda_copies >= 2,
            "auto_place_and_route should have inserted ≥2 Copy(_, Cuda) nodes; got {cuda_copies}"
        );
    }

    /// Full pipeline: recommend_placement → apply_placement_plan →
    /// fuel_graph::opt::insert_copies. Verifies Copy nodes appear at
    /// the boundary between CPU and CUDA placements.
    #[test]
    fn apply_then_insert_copies_emits_copies_at_device_boundaries() {
        // Hand-craft a dispatch table: size 12 → CPU, size 20 → CUDA.
        let entries = vec![
            ProfileEntry { op: OpKind::MatMul, dtype: DType::F32, size_class: SizeClass(12),
                backend: BackendId::Cpu,  device_index: 0, latency_ns:    10_000, iterations: 1, max_rel_error: 0.0 },
            ProfileEntry { op: OpKind::MatMul, dtype: DType::F32, size_class: SizeClass(12),
                backend: BackendId::Cuda, device_index: 0, latency_ns: 2_000_000, iterations: 1, max_rel_error: 0.0 },
            ProfileEntry { op: OpKind::MatMul, dtype: DType::F32, size_class: SizeClass(20),
                backend: BackendId::Cpu,  device_index: 0, latency_ns: 50_000_000, iterations: 1, max_rel_error: 0.0 },
            ProfileEntry { op: OpKind::MatMul, dtype: DType::F32, size_class: SizeClass(20),
                backend: BackendId::Cuda, device_index: 0, latency_ns:  5_000_000, iterations: 1, max_rel_error: 0.0 },
        ];
        let report = ProfileReport { version: PROFILE_REPORT_VERSION, entries };
        let table = DispatchTable::build(&report);

        // Build the heterogeneous graph (small + big matmul).
        let small_a = Tensor::from_f32(vec![0.0_f32; 64 * 64], Shape::from_dims(&[64, 64]), crate::Device::cpu().as_dyn());
        let small_b = small_a.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; 64 * 64]),
            Shape::from_dims(&[64, 64]),
        );
        let small_mm = small_a.matmul(&small_b);  // size 12 → CPU

        let big_a = small_a.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; 1024 * 1024]),
            Shape::from_dims(&[1024, 1024]),
        );
        let big_b = small_a.const_f32_like(
            Arc::<[f32]>::from(vec![0.0_f32; 1024 * 1024]),
            Shape::from_dims(&[1024, 1024]),
        );
        let big_mm = big_a.matmul(&big_b);  // size 20 → CUDA

        let n_before = small_a.graph().read().unwrap().len();

        let plan = recommend_placement(
            &small_a.graph().read().unwrap(),
            &table,
            Criterion::Fastest,
            DeviceLocation::Cpu,
        );
        apply_placement_plan(small_a.graph(), &plan);

        let roots = vec![small_mm.id(), big_mm.id()];
        let _new_roots = fuel_graph::opt::insert_copies(small_a.graph(), &roots);

        let g = small_a.graph().read().unwrap();
        let n_after = g.len();

        // big_a + big_b are placeless Const inputs feeding into big_mm
        // which is placed on CUDA → 2 Copy nodes get inserted.
        let mut cuda_copies = 0;
        for i in n_before..n_after {
            if let fuel_graph::Op::Copy { target } = &g.node(NodeId(i)).op {
                if matches!(target, DeviceLocation::Cuda { .. }) {
                    cuda_copies += 1;
                }
            }
        }
        assert!(cuda_copies >= 2,
            "expected ≥2 Copy(_, Cuda) nodes for big_a + big_b; got {cuda_copies} \
             (n_before={n_before} n_after={n_after})");
    }
}

//! Run extraction + the transient `lower_run` view + the fewness gate.
//!
//! Phase A PR-A2 of the "plan IS the graph" rebuild.
//!
//! A **run** is the maximal straight-line op-sequence between two
//! *decision points* — the future dispatch / CUDA-graph-capture unit
//! (see [`../docs/architecture/06-runtime.md`](../../docs/architecture/06-runtime.md)
//! §"Dispatch: runs, not single ops"). This module *derives* runs from
//! the graph structure; it does **not** execute them (Phase C) and does
//! **not** add an optimizer (Phase A3+). A run is a single-device
//! contiguous chain with one `entry` and one `exit`; its members are
//! [`NodeId`]s in topological order.
//!
//! Run boundaries (a new run starts at a node when *any* hold):
//! - **(a) graph entry** — a node with no predecessor inside the
//!   reachable set (a source / root input of the walk).
//! - **(b) reconverge** — a node named as some [`Op::Branch`]'s
//!   `reconverge_at` (the post-merge region is a fresh run).
//! - **(c) arm entry** — a node whose sole predecessor is a Branch's
//!   *diverge* point (the first node of each candidate route).
//! - **(d) residency seam** — the node's [`Graph::target_backend`]
//!   differs from its sole predecessor's, **or** the node is an
//!   [`Op::Copy`] / [`Op::Move`] device-transfer.
//! - **(e) fan-in** — a node with more than one predecessor inside the
//!   reachable set.
//!
//! Consequently a run never spans a `Branch` boundary or a device
//! change. A graph with **zero** `Op::Branch` nodes and a single
//! residency extracts to exactly **one** run covering all reachable
//! nodes — exactly today's single-route graph.

use crate::{Graph, NodeId, Op, topo_order_multi};
use fuel_ir::probe::BackendId;
use std::collections::{HashMap, HashSet};

/// A resolved route through the multi-path graph: for each `Op::Branch`
/// decision point (keyed by the Branch node's [`NodeId`]) the **arm
/// index** the runtime route picker (Picker 2) chose. Arm `i` is the
/// branch's `inputs[i]` (arm-0 = `inputs[0]` = the cost-vector winner).
///
/// A branch **absent** from the map defaults to **arm 0** — so the empty
/// route is exactly the arm-0-everywhere lowering ([`lower_runs_arm0`]),
/// which keeps the no-pressure / no-telemetry behavior identical to
/// Phase B. The picker in `fuel-dispatch` produces one of these by
/// walking the branches in topological order and consulting a runtime
/// selector over each branch's arms; `fuel-graph` only *consumes* it to
/// lower the chosen route (it has no selector / telemetry knowledge).
pub type PickedRoute = HashMap<NodeId, usize>;

/// A maximal straight-line, single-device op-sequence between two
/// decision points — the dispatch / command-buffer-capture unit.
///
/// `members` are in topological order; `entry == members[0]` and
/// `exit == *members.last()`. `device` is the resolved
/// [`BackendId`] shared by every member (via
/// [`Graph::target_backend`]); `None` means "inherit the executor
/// default" (no member carries an explicit backend) — a run is
/// single-device, so every member agrees on this value by construction
/// (a residency change is itself a run boundary).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Run {
    /// First node of the run (a decision-point boundary).
    pub entry: NodeId,
    /// Last node of the run (the node the next decision point reads,
    /// or a root output).
    pub exit: NodeId,
    /// Every node of the run, in topological order, `entry` first.
    pub members: Vec<NodeId>,
    /// The single resolved backend shared by all members, or `None`
    /// when no member carries an explicit `target_backend` (executor
    /// default).
    pub device: Option<BackendId>,
}

/// Extract the runs of the graph reachable from `root`, in
/// topological order of their entries.
///
/// Walks the single existing [`topo_order_multi`] order once and cuts
/// a new run at every boundary (see the module docs). Each emitted run
/// is single-entry / single-exit and single-device.
pub fn extract_runs(graph: &Graph, root: NodeId) -> Vec<Run> {
    extract_runs_multi(graph, &[root])
}

/// Multi-root variant of [`extract_runs`] — the reachable set is the
/// union over `roots` (mirrors [`topo_order_multi`]) **plus the arms of
/// every reachable [`Op::Branch`]**. A Branch's non-arm-0 arms are read
/// only by the Branch node itself (arm 0 is the runnability fallback the
/// `reconverge_at` node reads directly, per PR-A1), so seeding the walk
/// from the Branch nodes pulls every candidate route into the reachable
/// set.
pub fn extract_runs_multi(graph: &Graph, roots: &[NodeId]) -> Vec<Run> {
    let eff_roots = effective_roots(graph, roots);
    let order = topo_order_multi(graph, &eff_roots);
    let reachable: HashSet<NodeId> = order.iter().copied().collect();

    // Nodes named as some Branch's `reconverge_at` (the post-merge region
    // opens a fresh run).
    let reconverge_points: HashSet<NodeId> = order
        .iter()
        .filter_map(|&id| match graph.node(id).op {
            Op::Branch { reconverge_at } => Some(reconverge_at),
            _ => None,
        })
        .collect();

    // Arm-entry nodes: the first node of each candidate route, departing
    // from the shared diverge point. A run never spans into an arm.
    let arm_entries = compute_arm_entries(graph, &order, &reachable);

    // Predecessors inside the reachable set, per node (for fan-in + the
    // residency-seam sole-predecessor test).
    let mut pred_inside: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for &id in &order {
        let preds: Vec<NodeId> = graph
            .node(id)
            .inputs
            .iter()
            .copied()
            .filter(|p| reachable.contains(p))
            .collect();
        pred_inside.insert(id, preds);
    }

    let mut runs: Vec<Run> = Vec::new();
    let mut current: Vec<NodeId> = Vec::new();
    let mut current_device: Option<BackendId> = None;

    for &id in &order {
        // The `Op::Branch` (phi/merge) node is structural bookkeeping —
        // the decision point *between* runs, not executable work. It is
        // never a run member; once a route is picked it is invisible to
        // the hot path. Skipping it also means a consumer that reads a
        // Branch (rather than the `reconverge_at` node) finds its
        // predecessor absent from the open run and so opens a fresh run.
        if matches!(graph.node(id).op, Op::Branch { .. }) {
            continue;
        }
        let preds = &pred_inside[&id];
        let device = graph.target_backend(id);

        let starts_new = {
            // (a) graph entry — no predecessor in the reachable set.
            let is_entry = preds.is_empty();
            // (b) reconverge point.
            let is_reconverge = reconverge_points.contains(&id);
            // (c) arm entry.
            let is_arm = arm_entries.contains(&id);
            // (e) fan-in — more than one reachable predecessor.
            let is_fan_in = preds.len() > 1;
            // (d) residency seam: a device-transfer op, or a backend that
            // differs from the sole predecessor's backend.
            let is_transfer = matches!(graph.node(id).op, Op::Copy { .. } | Op::Move { .. });
            let is_seam = preds.len() == 1 && graph.target_backend(preds[0]) != device;
            // Contiguity: a run is a straight-line chain, so a node only
            // *extends* the current run when its sole predecessor IS the
            // current run's last member. Otherwise it begins a new run
            // even if no other rule fired (guards against topo interleave
            // and the Branch node itself, whose inputs are the arm exits).
            let breaks_chain = match current.last() {
                None => true,
                Some(&last) => !(preds.len() == 1 && preds[0] == last),
            };
            // A node whose backend differs from the open run's device
            // cannot extend it (single-device runs).
            let device_mismatch = !current.is_empty() && device != current_device;
            is_entry
                || is_reconverge
                || is_arm
                || is_fan_in
                || is_transfer
                || is_seam
                || breaks_chain
                || device_mismatch
        };

        if starts_new && !current.is_empty() {
            runs.push(finish_run(&current, current_device));
            current = Vec::new();
        }
        if current.is_empty() {
            current_device = device;
        }
        current.push(id);
    }
    if !current.is_empty() {
        runs.push(finish_run(&current, current_device));
    }
    runs
}

/// The effective root set for a run/density walk: the caller's `roots`
/// plus every [`Op::Branch`] node *participating in* the reachable
/// computation. A finalized `Op::Branch` is typically **orphaned** —
/// nothing downstream reads it (the `reconverge_at` node reads arm 0
/// directly, per PR-A1's runnability invariant), so it would never be
/// found by a plain forward walk. We therefore scan the whole arena and
/// seed any Branch whose `reconverge_at` (or any arm exit) is already
/// reachable; seeding it pulls its candidate arms into the walk. Run to
/// a fixpoint so a Branch reached only through another Branch's arm is
/// also covered.
///
/// Public so the optimizer's residency pass (`fuel-dispatch`) can stitch
/// inbound cross-device copies for EVERY surviving arm — a plain
/// `topo_order_multi(roots)` walk misses orphaned Branch nodes (and thus
/// their non-arm-0 arms), which would leave a re-pickable arm's device-
/// inputs un-bridged (Step E Phase C, PR C-0).
pub fn effective_roots(graph: &Graph, roots: &[NodeId]) -> Vec<NodeId> {
    let mut seeds: Vec<NodeId> = roots.to_vec();
    let mut seen: HashSet<NodeId> = seeds.iter().copied().collect();
    loop {
        let reachable: HashSet<NodeId> = topo_order_multi(graph, &seeds).into_iter().collect();
        let mut added = false;
        for idx in 0..graph.len() {
            let id = NodeId(idx);
            if seen.contains(&id) {
                continue;
            }
            let Op::Branch { reconverge_at } = graph.node(id).op else { continue };
            // The branch participates if its merge target or any arm exit
            // is already part of the reachable computation.
            let participates = reachable.contains(&reconverge_at)
                || graph.node(id).inputs.iter().any(|a| reachable.contains(a));
            if participates {
                seen.insert(id);
                seeds.push(id);
                added = true;
            }
        }
        if !added {
            break;
        }
    }
    seeds
}

/// The reachable [`Op::Branch`] decision points over `roots`, in
/// **topological order** of their `reconverge_at` merge points — the
/// order the runtime route picker (Picker 2) resolves them in, so that
/// coupled upstream decisions are committed before a downstream branch
/// is reached (06-runtime §"Resolution order matters when decisions are
/// coupled").
///
/// A finalized `Op::Branch` is typically orphaned (the `reconverge_at`
/// node reads arm 0 directly, per PR-A1), so a plain forward walk would
/// miss it; we therefore order by each branch's `reconverge_at` position
/// in the [`effective_roots`] walk — the merge point is downstream of the
/// arms, so its topo position is a faithful "where this decision sits"
/// key. Branches whose merge point is unreachable are dropped.
pub fn branches_in_topo_order(graph: &Graph, roots: &[NodeId]) -> Vec<NodeId> {
    let eff_roots = effective_roots(graph, roots);
    let order = topo_order_multi(graph, &eff_roots);
    let position: HashMap<NodeId, usize> =
        order.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    let mut branches: Vec<(usize, NodeId)> = Vec::new();
    for idx in 0..graph.len() {
        let id = NodeId(idx);
        let Op::Branch { reconverge_at } = graph.node(id).op else {
            continue;
        };
        // Only branches whose merge point participates in this realize.
        if let Some(&pos) = position.get(&reconverge_at) {
            branches.push((pos, id));
        }
    }
    branches.sort_by_key(|&(pos, _)| pos);
    branches.into_iter().map(|(_, id)| id).collect()
}

fn finish_run(members: &[NodeId], device: Option<BackendId>) -> Run {
    Run {
        entry: members[0],
        exit: *members.last().expect("a run has at least one member"),
        members: members.to_vec(),
        device,
    }
}

/// For every [`Op::Branch`] reachable in `order`, find the arm-entry
/// nodes: the first node on each candidate route after the shared
/// diverge point. These are run boundaries (rule (c)).
///
/// The op carries `reconverge_at` but not the diverge, so we recover the
/// branch's shared prefix structurally: intersect the backward cones of
/// all arm exits. Every node in that intersection is shared (the diverge
/// point and its ancestors); the per-arm interior is everything else in
/// the cone. An interior node is an *arm entry* when one of its
/// predecessors lies in the shared prefix — i.e. it is the point where
/// the route departs from the diverge region.
fn compute_arm_entries(
    graph: &Graph,
    order: &[NodeId],
    reachable: &HashSet<NodeId>,
) -> HashSet<NodeId> {
    let mut arm_entries: HashSet<NodeId> = HashSet::new();
    for &id in order {
        let Op::Branch { .. } = graph.node(id).op else { continue };
        let arm_exits: Vec<NodeId> = graph.node(id).inputs.clone();
        if arm_exits.len() < 2 {
            continue;
        }
        // Backward cone of each arm exit, bounded to the reachable set.
        let cones: Vec<HashSet<NodeId>> = arm_exits
            .iter()
            .map(|&e| backward_cone(graph, e, reachable))
            .collect();
        // The diverge point is shared by every arm — it lies in the
        // intersection of all cones. The arm interiors are the
        // per-cone nodes NOT in that shared prefix. The arm entry is the
        // interior node whose sole predecessor lies in the shared prefix.
        let mut shared: HashSet<NodeId> = cones[0].clone();
        for c in &cones[1..] {
            shared = shared.intersection(c).copied().collect();
        }
        for cone in &cones {
            for &n in cone {
                if shared.contains(&n) {
                    continue;
                }
                // Arm-interior node. It is an arm *entry* if any of its
                // predecessors is in the shared prefix (it departs from
                // the diverge region).
                let departs = graph
                    .node(n)
                    .inputs
                    .iter()
                    .any(|p| shared.contains(p));
                if departs {
                    arm_entries.insert(n);
                }
            }
        }
    }
    arm_entries
}

/// Backward-reachable cone of `from` (the node and all transitive
/// inputs), bounded to `reachable`.
fn backward_cone(graph: &Graph, from: NodeId, reachable: &HashSet<NodeId>) -> HashSet<NodeId> {
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut stack = vec![from];
    while let Some(n) = stack.pop() {
        if !reachable.contains(&n) || !seen.insert(n) {
            continue;
        }
        for &inp in &graph.node(n).inputs {
            if !seen.contains(&inp) {
                stack.push(inp);
            }
        }
    }
    seen
}

/// The transient executable view of a single run: its ordered member
/// [`NodeId`]s, ready to be recorded into a command buffer (Phase C).
///
/// A **borrow / transient** view — it is NOT stored on the graph. This
/// is the shape Phase C records; PR-A2 only derives it.
pub fn lower_run(run: &Run) -> &[NodeId] {
    &run.members
}

/// The **arm-0 single-route lowering** (Phase A PR-A4, architect-
/// approved temporary fix). The flat executable dispatch order that
/// **follows arm 0 through every [`Op::Branch`]** — the pre-run, arm 0's
/// run, then the post-run — and **skips every non-arm-0 arm's run**.
///
/// Why this is needed: [`extract_runs_multi`] partitions *every* arm
/// into its own run (so the future runtime picker can choose among
/// them). A naive concatenation of all runs would execute every arm.
/// Until the Phase-C route picker lands, realize must default to arm 0
/// — the route a finalized-but-unpicked graph runs on (the A1 arm-0
/// runnability invariant). This walk drops the runs that lie entirely
/// inside a non-arm-0 arm and keeps the rest (pre / arm-0 / post), so a
/// branched graph realizes to the same result as its arm-0 route.
///
/// A graph with **zero `Op::Branch` nodes** has no arm to skip, so this
/// is byte-identical to concatenating [`lower_run`] over
/// [`extract_runs_multi`] — today's single-route order.
pub fn lower_runs_arm0(graph: &Graph, roots: &[NodeId]) -> Vec<NodeId> {
    // Arm-0 everywhere is the empty-route special case of the general
    // route-aware lowering (an unmentioned branch defaults to arm 0).
    lower_picked_route(graph, roots, &PickedRoute::new())
}

/// **Route-aware lowering** (Phase C PR-C1) — the generalization of
/// [`lower_runs_arm0`]. The flat executable dispatch order that follows
/// the **chosen arm at each [`Op::Branch`]** (per `picked`, defaulting to
/// arm 0 for any branch the picker did not resolve) and **skips every
/// non-chosen arm's run**.
///
/// This is exactly [`lower_runs_arm0`] generalized from "arm 0 always"
/// to "the picked arm per branch": it reuses the same run partition
/// ([`extract_runs_multi`]) and the same single-contiguous-region
/// property of a run (extract never spans a branch boundary, so a run is
/// either wholly inside a non-chosen arm or wholly outside). The skip set
/// is [`non_chosen_arm_nodes`] — `non_arm0_arm_nodes` generalized to "any
/// arm but the chosen one."
///
/// Value-preserving contract: every arm is a valid kernel for the same
/// op (cast-to-uniform at `reconverge_at` per PR-A1), so following any
/// arm yields the same result within tolerance. With `picked` empty
/// (no-pressure / no-telemetry route) this is byte-identical to
/// [`lower_runs_arm0`], and a graph with **zero `Op::Branch` nodes** has
/// no arm to skip, so it equals concatenating [`lower_run`] over
/// [`extract_runs_multi`] — today's single-route order.
pub fn lower_picked_route(
    graph: &Graph,
    roots: &[NodeId],
    picked: &PickedRoute,
) -> Vec<NodeId> {
    let runs = extract_runs_multi(graph, roots);
    let skip = non_chosen_arm_nodes(graph, roots, picked);
    // The KEPT runs (chosen arms + pre/post/independent regions), in topo
    // order — the runs the un-reordered lowering would concatenate.
    let kept: Vec<Run> = runs
        .into_iter()
        // A run is a single contiguous arm/region (extract never spans a
        // branch boundary), so it is either wholly inside a non-chosen
        // arm or wholly outside. Keep iff its entry is not a skipped node.
        .filter(|run| !skip.contains(&run.entry))
        .collect();
    // PR-C3: device-alternating reorder of the kept runs (auto-overlap).
    // Single-device ⇒ identity ⇒ byte-identical to the prior concatenation.
    // Multi-device ⇒ a valid topological reordering that interleaves
    // devices so independent device sub-DAGs dispatch adjacently (the A4b
    // overlap topology, for an arbitrary graph). The reorder is a hint: the
    // emitted NodeIds are identical, only their order across independent
    // runs changes.
    let perm = device_alternating_order(graph, &kept);
    let mut order = Vec::new();
    for &ri in &perm {
        order.extend_from_slice(lower_run(&kept[ri]));
    }
    order
}

/// **Device-overlap topological reorder** of a run list (Step E Phase C,
/// PR C3 — the *auto-overlap* pass). Returns a permutation of
/// `0..runs.len()` (indices into `runs`) that is a **valid topological
/// reordering** — every run still appears after all the runs it depends
/// on — chosen so independent cross-device sub-DAGs DISPATCH so they
/// overlap: the longest compute chunk of each device is enqueued/recorded
/// BEFORE the host-blocking cross-device drain that feeds the join.
///
/// # Why (the A4b caveat this removes)
///
/// [`extract_runs_multi`] emits runs in [`topo_order_multi`]'s fixed-DFS
/// order. For two INDEPENDENT reconverging sub-DAGs — one on CUDA, one on
/// Vulkan — that DFS emits one sub-DAG *fully* (including the cross-device
/// `Op::Copy` that feeds the join) before the other is even recorded. The
/// executor then blocks the host on that copy's producer (a D2H wait)
/// while the OTHER device sits idle — the two devices serialize. A4b built
/// the cross-device concurrency MECHANISM (eager Vulkan submit + in-flight
/// CUDA handles) but it only overlaps when the auto-submitting device's
/// chunk (CUDA, A3 stream-ordered) is enqueued — and thus running — BEFORE
/// the host blocks on the deferred device's (Vulkan) drain. The A4b overlap
/// benchmark gets that order by HAND (a specific reconverge input order +
/// CPU-primary pinning so the DFS happens to pop the heavy CUDA chunk
/// first). This pass produces that order for an ARBITRARY graph.
///
/// # The heuristic — longest-downstream-compute (critical-path) list scheduling
///
/// A Kahn pump that, among the runs whose producers are all emitted, picks
/// the run with the **largest compute chunk reachable downstream**
/// (including itself) — a classic critical-path / HLFET list-scheduling
/// priority. Concretely, each run is weighted by `max over its downstream
/// cone of compute_size(run)`, where a compute run's size is its node count
/// and a cross-device **transfer** (`Op::Copy`/`Op::Move`) has size 0. The
/// effect:
///
/// - The path to the **heaviest compute chunk** (e.g. the long CUDA chain)
///   is emitted FIRST — so on the auto-submitting device it is enqueued and
///   running earliest, maximizing the window the other device's work
///   overlaps. (Empirically: enqueuing the heavy CUDA chunk first is what
///   makes the eager-submit overlap; the inverse order serializes.)
/// - The next-heaviest device chunk is emitted next (its device's chunk is
///   recorded while the first device runs).
/// - A cross-device **drain** (the D2H `Op::Copy` feeding the join) has
///   near-zero downstream compute weight, so it sorts AFTER both device
///   chunks — exactly the A4b overlap topology (record/enqueue both chunks,
///   THEN the drain whose fence-wait the other device's in-flight work
///   hides). Among equal-weight ready runs, ties break toward a device
///   DIFFERENT from the last emitted (interleave), then the smallest
///   original index (determinism).
///
/// # The invariant — a HINT, never a correctness lever
///
/// Reordering INDEPENDENT runs changes only *when* work dispatches, never
/// the result: the emitted order is always a valid topological order (a
/// run after its producers), every op is bit-deterministic, and every
/// cross-device wait/copy keys on a node's *inputs*, not on dispatch
/// position — so no reorder can drop a wait or change a value. The output
/// is byte-identical regardless of the reorder (proved by the
/// reorder-invariance tests).
///
/// # The multi-device gate (single-device = identity)
///
/// If every run shares a single resolved device (≤ 1 distinct
/// [`Run::device`] across the list), there is nothing to overlap, so this
/// returns the identity permutation `0..runs.len()` and the lowering is
/// **byte-identical to today's** topo-DFS order. The pass only diverges
/// from the input order when the run list genuinely spans two or more
/// devices.
///
/// # Determinism
///
/// Deterministic for a given run list: the priority key is
/// `(−downstream_weight, not_different_device, original_index)`, fully
/// ordered, so the reorder is reproducible and a degenerate input (one
/// device) reproduces the input order exactly.
///
/// `runs` must be the partition [`extract_runs_multi`] produced for the
/// same `graph` (the node→run mapping is rebuilt from `runs`), and the
/// dependency edges are derived from each member's graph inputs.
pub fn device_alternating_order(graph: &Graph, runs: &[Run]) -> Vec<usize> {
    let n = runs.len();
    if n <= 1 {
        return (0..n).collect();
    }

    // The multi-device gate: count distinct resolved devices. `None`
    // (executor-default, no explicit backend) is its own bucket — a list
    // that is all-`None` or all-one-backend has nothing to interleave.
    let mut distinct: Vec<Option<BackendId>> = Vec::new();
    for r in runs {
        if !distinct.contains(&r.device) {
            distinct.push(r.device);
        }
    }
    if distinct.len() <= 1 {
        // Single-device (or single executor-default) — identity. This is
        // the byte-identical-to-today path (the headline single-device
        // regression gate).
        return (0..n).collect();
    }

    // node -> the index of the run that contains it (every reachable node
    // is in exactly one run; the Branch node is not a member of any run,
    // so a cross-run input that is a Branch node simply finds no producer
    // run, which is correct — arms feed the reconverge, not the Branch).
    let mut run_of: HashMap<NodeId, usize> = HashMap::new();
    for (ri, r) in runs.iter().enumerate() {
        for &m in &r.members {
            run_of.insert(m, ri);
        }
    }

    // Run-level dependency edges. Run `r` depends on run `s` (s != r) when
    // some member of `r` reads a node that is a member of `s` (a cross-run
    // input). `preds[r]` = the set of producer runs; `succs[s]` = the runs
    // that read `s`; `indeg[r]` = count of not-yet-emitted producers.
    let mut preds: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (ri, r) in runs.iter().enumerate() {
        for &m in &r.members {
            for &inp in &graph.node(m).inputs {
                if let Some(&pi) = run_of.get(&inp) {
                    if pi != ri && preds[ri].insert(pi) {
                        succs[pi].push(ri);
                    }
                }
            }
        }
    }
    let mut indeg: Vec<usize> = preds.iter().map(|p| p.len()).collect();

    // Per-run COMPUTE size: node count for a real compute chunk, 0 for a
    // pure cross-device TRANSFER (`Op::Copy`/`Op::Move` — always its own
    // single-member run, the residency-seam boundary). A transfer is the
    // run that FORCES a cross-device wait at dispatch (the executor drains
    // the source / eager-submits + fences the in-flight batch when it hits a
    // cross-device copy); it does no compute, so it weighs 0 and a drain is
    // never the heaviest-downstream pick — it sorts after the compute chunks.
    let compute_size = |ri: usize| -> u64 {
        if matches!(graph.node(runs[ri].entry).op, Op::Copy { .. } | Op::Move { .. }) {
            0
        } else {
            runs[ri].members.len() as u64
        }
    };

    // `down_weight[r]` = the LARGEST compute chunk reachable from `r`
    // downstream, including `r` itself — the critical-path priority. A
    // memoized DFS over `succs` (the run-DAG is acyclic, so the recursion
    // terminates; memoization makes it linear). Computed with an explicit
    // stack (never-panic on deep graphs — no recursion).
    let mut down_weight: Vec<u64> = vec![u64::MAX; n]; // MAX = not yet computed
    for start in 0..n {
        if down_weight[start] != u64::MAX {
            continue;
        }
        // Post-order: push (node, children_done?).
        let mut stack: Vec<(usize, bool)> = vec![(start, false)];
        while let Some((r, done)) = stack.pop() {
            if done {
                let mut w = compute_size(r);
                for &s in &succs[r] {
                    w = w.max(down_weight[s]);
                }
                down_weight[r] = w;
                continue;
            }
            if down_weight[r] != u64::MAX {
                continue;
            }
            // Tentatively mark in-progress so a (DAG-impossible) revisit
            // doesn't loop; finalized on the `done` pass.
            stack.push((r, true));
            for &s in &succs[r] {
                if down_weight[s] == u64::MAX {
                    stack.push((s, false));
                }
            }
        }
    }

    // Kahn pump with critical-path priority. `ready` is every run whose
    // producers are all emitted. Among the ready runs we pick the one whose
    // priority key is smallest:
    //
    //   ( -down_weight ,  not_different_device ,  original_index )
    //
    // i.e. (1) the LARGEST downstream compute chunk first — emit the path to
    // the heaviest device chunk earliest so the auto-submitting device
    // (CUDA) is enqueued + running before the deferred device's (Vulkan)
    // host-blocking drain, which is what the eager-submit mechanism turns
    // into overlap (and a drain weighs 0, so it sorts after both chunks);
    // (2) among equal-weight ready runs, a device DIFFERENT from the last
    // emitted (interleave); (3) the SMALLEST original index (determinism +
    // topo stability — the degenerate single-device case already returned
    // identity above). `ready` is a plain Vec scanned each step; run counts
    // are tiny (a run is a whole straight-line chunk; the fewness gate keeps
    // decision points sparse), so the O(ready) scan is cheap. This is a
    // HINT: every emitted order is a valid topological order regardless of
    // the priority (Kahn guarantees producers-first), so the result is
    // byte-identical — only the dispatch timing changes.
    let mut emitted: Vec<bool> = vec![false; n];
    let mut ready: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut last_device: Option<Option<BackendId>> = None;

    while !ready.is_empty() {
        // Priority key (smaller wins): heaviest downstream compute first,
        // then a different device than last (interleave), then smallest idx.
        let key_of = |ri: usize| -> (std::cmp::Reverse<u64>, bool, usize) {
            let same_device =
                last_device.map_or(false, |ld| runs[ri].device == ld);
            (std::cmp::Reverse(down_weight[ri]), same_device, ri)
        };
        let mut best_pos = 0usize;
        let mut best_key = key_of(ready[0]);
        for (pos, &ri) in ready.iter().enumerate().skip(1) {
            let key = key_of(ri);
            if key < best_key {
                best_key = key;
                best_pos = pos;
            }
        }
        let ri = ready.swap_remove(best_pos);
        emitted[ri] = true;
        order.push(ri);
        last_device = Some(runs[ri].device);

        for &si in &succs[ri] {
            if emitted[si] {
                continue;
            }
            indeg[si] -= 1;
            if indeg[si] == 0 {
                ready.push(si);
            }
        }
    }

    // A cyclic run graph is impossible (the node graph is a DAG and runs
    // partition it), so the pump emits every run. Defend never-panic: if
    // somehow a run was not emitted (a structural surprise), fall back to
    // the input order for the remainder so the lowering stays total rather
    // than dropping work.
    if order.len() != n {
        for i in 0..n {
            if !emitted[i] {
                order.push(i);
            }
        }
    }
    order
}

/// **Streaming route-aware lowering** (Step E Phase C, PR C1) — the
/// incremental form of [`lower_picked_route`]. Instead of taking a
/// fully-resolved [`PickedRoute`] up front, it walks the run partition in
/// topological order and **resolves each branch lazily, the first time the
/// walk reaches one of its arm-entry runs**, by calling `resolve(branch)`.
/// It emits only the chosen arm's runs (skipping the non-chosen arms),
/// exactly like [`lower_picked_route`] — but the arm decision happens *at
/// the frontier*, not before the walk.
///
/// This is the substrate that lets the runtime picker re-pick a branch's
/// arm **by the live device load at the moment the frontier reaches it**
/// (C2): the compiler thread drives this walk while the executor drains
/// the previous runs, so `resolve` sees the load current to that decision
/// point. `resolve` returns the chosen arm index for a branch (a stale /
/// out-of-range index is clamped to arm 0 by the same skip logic, so it
/// never panics — never-panic on a production path).
///
/// **Byte-identity with the one-shot lowering (the C1 gate).** Because
/// `resolve` is the same per-branch decision the eager picker makes (the
/// production VRAM-pressure chain reads only free-memory state, not walk
/// progress), the route this walk accumulates equals
/// [`fuel_graph::PickedRoute`] the eager `pick_route` produces, and the
/// emitted order therefore equals `lower_picked_route(graph, roots,
/// pick_route(..))` on every input. C1 changes *when* a branch resolves,
/// never *which* arm — and a branchless graph never enters this walk at
/// all (the caller's branchless fast path returns before streaming).
/// `resolve` is invoked **at most once per branch** (memoized in the
/// accumulated route), so a coupled downstream branch that reads upstream
/// picks still resolves upstream-first (arm-entries of an upstream branch
/// precede a downstream branch's in run-topo order).
pub fn lower_picked_route_streaming<F>(
    graph: &Graph,
    roots: &[NodeId],
    mut resolve: F,
) -> Vec<NodeId>
where
    F: FnMut(NodeId) -> usize,
{
    let mut order = Vec::new();
    walk_picked_route_streaming(graph, roots, &mut resolve, |members| {
        order.extend_from_slice(members);
    });
    order
}

/// The callback core of the streaming lowering — the form the compiler
/// thread drives so resolution and emission genuinely interleave (the
/// thread compiles + dispatches each chosen run as the walk reaches it,
/// rather than first materializing a flat order). `resolve(branch)` yields
/// the chosen arm index at the frontier; `emit(&[NodeId])` receives each
/// kept run's members in topological order. [`lower_picked_route_streaming`]
/// is the `Vec`-collecting wrapper over this (used by the byte-identity
/// test); the executor's compiler thread passes an `emit` that
/// `compile_one`'s + sends each node.
///
/// Walks the run partition once, in topological order of run entries
/// (identical to [`extract_runs_multi`]); at the first run that opens an
/// arm of a not-yet-resolved branch it calls `resolve`, folds that
/// branch's non-chosen arms into the skip set, then emits the run iff its
/// entry is not skipped. See [`lower_picked_route_streaming`] for the
/// byte-identity argument.
pub fn walk_picked_route_streaming<R, E>(
    graph: &Graph,
    roots: &[NodeId],
    mut resolve: R,
    mut emit: E,
) where
    R: FnMut(NodeId) -> usize,
    E: FnMut(&[NodeId]),
{
    let runs = extract_runs_multi(graph, roots);

    // Per-branch lowering metadata, precomputed once from the reachable
    // structure (cones + the diverge/shared prefix + each arm's entry
    // node). The *decision* — which arm — is still deferred to `resolve`
    // at the frontier; only the structure is precomputed.
    let branch_meta = branch_arm_meta(graph, roots);

    // arm-entry node -> the branches it opens an arm of. A node can be the
    // arm-entry of more than one branch only in degenerate shared-prefix
    // overlaps; resolving every such branch on first encounter is correct
    // (each resolves independently) and keeps the walk single-pass.
    let mut entry_to_branches: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for meta in &branch_meta {
        for &(entry, _arm) in &meta.arm_entries {
            entry_to_branches.entry(entry).or_default().push(meta.branch);
        }
    }
    let meta_of: HashMap<NodeId, &BranchArmMeta> =
        branch_meta.iter().map(|m| (m.branch, m)).collect();

    let mut skip: HashSet<NodeId> = HashSet::new();
    let mut resolved: HashSet<NodeId> = HashSet::new();

    // PR-C3 composition: the device-alternating reorder (auto-overlap) runs
    // over the kept runs, but it must NOT reorder across an unresolved
    // branch decision (a downstream run's keep/skip depends on the arm
    // picked upstream, and the picker reads live load AT the frontier, so
    // moving a run past an unresolved branch would change what is even
    // resolvable). So we reorder within **branch-free segments**: buffer
    // kept runs; when the frontier reaches a run that resolves a NEW branch,
    // FLUSH the buffered segment (device-alternation-reordered) BEFORE
    // resolving, then start a fresh segment. The flush is the only place a
    // reorder happens, and it never crosses a `resolve` call — so arm
    // resolution stays strictly upstream-first and the reorder is confined
    // to runs already committed to their devices. Branchless graphs (the
    // auto-overlap benchmark + every existing multi-device live suite) have
    // exactly one segment = the whole run list, so this is identical to the
    // one-shot `lower_picked_route` reorder.
    let mut segment: Vec<&Run> = Vec::new();
    let flush = |segment: &mut Vec<&Run>, emit: &mut E| {
        if segment.is_empty() {
            return;
        }
        // `device_alternating_order` wants `&[Run]`; collect the segment's
        // runs (cheap — members are NodeId Vecs, but a segment is small).
        let owned: Vec<Run> = segment.iter().map(|r| (*r).clone()).collect();
        let perm = device_alternating_order(graph, &owned);
        for &ri in &perm {
            emit(lower_run(&owned[ri]));
        }
        segment.clear();
    };

    for run in &runs {
        // Frontier reached this run's entry. If it opens an arm of one or
        // more not-yet-resolved branches, the route past this point is about
        // to be decided — flush the branch-free segment accumulated so far
        // (reordered) BEFORE resolving, so no reorder crosses the decision.
        if let Some(branches) = entry_to_branches.get(&run.entry) {
            let opens_new = branches.iter().any(|b| !resolved.contains(b));
            if opens_new {
                flush(&mut segment, &mut emit);
            }
            for &branch in branches {
                if !resolved.insert(branch) {
                    continue;
                }
                let meta = meta_of[&branch];
                // Resolve NOW (reading live load via `resolve` once C2 lands)
                // and fold this branch's non-chosen arms into the skip set —
                // before we decide whether to keep this run.
                let chosen = resolve(branch);
                add_non_chosen_skip(meta, chosen, &mut skip);
            }
        }
        // A run is a single contiguous arm/region (extract never spans a
        // branch boundary), so it is either wholly inside a non-chosen arm
        // or wholly outside. Skip iff its entry is a skipped node; otherwise
        // buffer it into the current branch-free segment (emitted at the
        // next flush, device-alternation-reordered).
        if skip.contains(&run.entry) {
            continue;
        }
        segment.push(run);
    }
    // Flush the final branch-free segment (the post-region + any tail).
    flush(&mut segment, &mut emit);
}

/// Per-branch structural metadata for the streaming lowering: the branch
/// node, its arms' `(arm-entry node, arm index)` pairs, the per-arm
/// backward cones, and the shared diverge prefix. The arm *decision* is
/// not here — only the structure the skip-set computation needs once an
/// arm is chosen.
struct BranchArmMeta {
    branch: NodeId,
    /// `(arm-entry node, arm index)` for each arm — the run entry that
    /// opens that arm. Used to detect, while walking runs, which branch a
    /// run opens an arm of.
    arm_entries: Vec<(NodeId, usize)>,
    /// Backward cone of each arm exit (`inputs[i]`), bounded to reachable.
    cones: Vec<HashSet<NodeId>>,
    /// Shared diverge prefix = intersection of every arm's cone.
    shared: HashSet<NodeId>,
}

/// Precompute [`BranchArmMeta`] for every reachable ≥2-arm [`Op::Branch`].
/// Mirrors `non_chosen_arm_nodes`'s cone/shared-prefix derivation and
/// `compute_arm_entries`'s arm-entry recovery (the op carries only
/// `reconverge_at`, so the diverge prefix is recovered as the cone
/// intersection), so the streaming skip set is byte-identical to the
/// one-shot `non_chosen_arm_nodes` for the same picks.
fn branch_arm_meta(graph: &Graph, roots: &[NodeId]) -> Vec<BranchArmMeta> {
    let eff_roots = effective_roots(graph, roots);
    let order = topo_order_multi(graph, &eff_roots);
    let reachable: HashSet<NodeId> = order.iter().copied().collect();

    let mut metas = Vec::new();
    for &id in &order {
        let Op::Branch { .. } = graph.node(id).op else { continue };
        let arm_exits = graph.node(id).inputs.clone();
        if arm_exits.len() < 2 {
            continue;
        }
        let cones: Vec<HashSet<NodeId>> = arm_exits
            .iter()
            .map(|&e| backward_cone(graph, e, &reachable))
            .collect();
        let mut shared: HashSet<NodeId> = cones[0].clone();
        for c in &cones[1..] {
            shared = shared.intersection(c).copied().collect();
        }
        // Each arm's entry: the interior node (cone minus shared) whose
        // sole/any predecessor lies in the shared prefix — where the arm
        // departs from the diverge region (mirrors `compute_arm_entries`).
        let mut arm_entries: Vec<(NodeId, usize)> = Vec::new();
        for (arm_idx, cone) in cones.iter().enumerate() {
            for &n in cone {
                if shared.contains(&n) {
                    continue;
                }
                let departs =
                    graph.node(n).inputs.iter().any(|p| shared.contains(p));
                if departs {
                    arm_entries.push((n, arm_idx));
                }
            }
        }
        metas.push(BranchArmMeta { branch: id, arm_entries, cones, shared });
    }
    metas
}

/// Fold one branch's non-chosen arms into `skip`, given the chosen arm
/// index. Identical to the per-branch body of [`non_chosen_arm_nodes`]: a
/// node is skipped when it lies in a non-chosen arm's cone but is neither
/// shared nor part of the chosen arm's cone. A stale / out-of-range
/// `chosen` clamps to arm 0 (never panic).
fn add_non_chosen_skip(
    meta: &BranchArmMeta,
    chosen: usize,
    skip: &mut HashSet<NodeId>,
) {
    let chosen = if chosen < meta.cones.len() { chosen } else { 0 };
    let chosen_cone = &meta.cones[chosen];
    for (i, cone) in meta.cones.iter().enumerate() {
        if i == chosen {
            continue;
        }
        for &n in cone {
            if !meta.shared.contains(&n) && !chosen_cone.contains(&n) {
                skip.insert(n);
            }
        }
    }
}

/// The set of nodes that belong to a **non-chosen arm** of some reachable
/// [`Op::Branch`] — the nodes the route-aware lowering skips. The arm-0
/// generalization: when `picked` is empty (or a branch is absent from
/// it), the chosen arm defaults to arm 0, so this reduces to the former
/// `non_arm0_arm_nodes`.
///
/// For each reachable branch, the chosen arm is `inputs[chosen]`
/// (`chosen = picked[branch]`, default 0) and the non-chosen arms are
/// every other input. The shared prefix (the diverge region) is the
/// intersection of every arm exit's backward cone (PR-A2's
/// `compute_arm_entries` recovers the diverge the same way — the op
/// carries `reconverge_at`, not the diverge). A node is skipped when it
/// lies in a non-chosen arm's cone but is neither shared nor part of the
/// chosen arm's cone — i.e. it is interior to a route the picker did not
/// take. Arms are internally disjoint by the PR-A1 build-time
/// validation, so these sets don't overlap the chosen arm.
fn non_chosen_arm_nodes(
    graph: &Graph,
    roots: &[NodeId],
    picked: &PickedRoute,
) -> HashSet<NodeId> {
    let eff_roots = effective_roots(graph, roots);
    let order = topo_order_multi(graph, &eff_roots);
    let reachable: HashSet<NodeId> = order.iter().copied().collect();

    let mut skip: HashSet<NodeId> = HashSet::new();
    for &id in &order {
        let Op::Branch { .. } = graph.node(id).op else { continue };
        let arm_exits = &graph.node(id).inputs;
        if arm_exits.len() < 2 {
            continue;
        }
        // The chosen arm index — default arm 0 (the winner) for any
        // branch the picker did not resolve. Clamp to a valid arm so a
        // stale/out-of-range pick degrades to arm 0 rather than panicking
        // (never panic on a production path).
        let chosen = picked
            .get(&id)
            .copied()
            .filter(|&c| c < arm_exits.len())
            .unwrap_or(0);

        let cones: Vec<HashSet<NodeId>> = arm_exits
            .iter()
            .map(|&e| backward_cone(graph, e, &reachable))
            .collect();
        // Shared prefix = intersection of every arm's cone (the diverge
        // region the arms depart from).
        let mut shared: HashSet<NodeId> = cones[0].clone();
        for c in &cones[1..] {
            shared = shared.intersection(c).copied().collect();
        }
        // Skip every non-chosen arm's interior: its cone minus the shared
        // prefix minus the chosen arm's cone.
        let chosen_cone = &cones[chosen];
        for (i, cone) in cones.iter().enumerate() {
            if i == chosen {
                continue;
            }
            for &n in cone {
                if !shared.contains(&n) && !chosen_cone.contains(&n) {
                    skip.insert(n);
                }
            }
        }
    }
    skip
}

/// Branch density: the share of reachable nodes that are decision
/// points ([`Op::Branch`] nodes), over `root`.
///
/// The **fewness gate** — deliberate-fork branching must stay sparse
/// (the granularity crux locked before Phase B). A real decode graph
/// branches only at "layer" boundaries and sits far below the threshold;
/// a synthetic per-op-branch graph blows past it.
pub fn branch_density(graph: &Graph, root: NodeId) -> f32 {
    branch_density_multi(graph, &[root])
}

/// Multi-root variant of [`branch_density`].
pub fn branch_density_multi(graph: &Graph, roots: &[NodeId]) -> f32 {
    let eff_roots = effective_roots(graph, roots);
    let order = topo_order_multi(graph, &eff_roots);
    if order.is_empty() {
        return 0.0;
    }
    let branches = order
        .iter()
        .filter(|&&id| matches!(graph.node(id).op, Op::Branch { .. }))
        .count();
    branches as f32 / order.len() as f32
}

/// The fewness-gate threshold: branch points must be at most ~5% of
/// reachable nodes. Deliberate-fork branching stays this sparse;
/// per-op branching does not.
pub const FEWNESS_THRESHOLD: f32 = 0.05;

/// Whether the graph passes the fewness gate over `root` (branch
/// density strictly below [`FEWNESS_THRESHOLD`]).
pub fn passes_fewness_gate(graph: &Graph, root: NodeId) -> bool {
    branch_density(graph, root) < FEWNESS_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Node;
    use fuel_ir::{DType, Shape};

    fn f32_node(g: &mut Graph, op: Op, inputs: Vec<NodeId>) -> NodeId {
        g.push(Node {
            op,
            inputs,
            shape: Shape::from_dims(&[2]),
            dtype: DType::F32,
        })
    }

    /// Hand-build the same 2-arm diamond the branch-builder tests use,
    /// then finalize it into a real `Op::Branch`. Returns the graph and
    /// `(diverge, arm0, arm1, reconverge, branch)`.
    ///
    /// Topology: `diverge -> {arm0, arm1}`; `reconverge` reads arm0 (the
    /// runnability invariant); a `Branch` node merges {arm0, arm1} with
    /// `reconverge_at = reconverge`. `post` reads `reconverge` so the
    /// post-merge region is non-empty.
    fn diamond_with_branch() -> (Graph, NodeId, NodeId, NodeId, NodeId, NodeId, NodeId) {
        let mut g = Graph::new();
        let pre = f32_node(&mut g, Op::Const, vec![]);
        let diverge = f32_node(&mut g, Op::Relu, vec![pre]);
        let arm0 = f32_node(&mut g, Op::Silu, vec![diverge]);
        let arm1 = f32_node(&mut g, Op::Gelu, vec![diverge]);
        // reconverge reads arm0 (arm-0 runnability).
        let reconverge = f32_node(&mut g, Op::Relu, vec![arm0]);
        let mut b = g.open_branch(diverge);
        b.add_arm(arm0);
        b.add_arm(arm1);
        let branch = b
            .finalize_branches(&mut g, reconverge)
            .expect("well-formed 2-arm branch finalizes")
            .expect("2 arms survive");
        // A post node that reads the merge, so there is a real post-run.
        let post = f32_node(&mut g, Op::Tanh, vec![reconverge]);
        (g, diverge, arm0, arm1, reconverge, branch, post)
    }

    /// (a) A straight-line graph with no branches extracts to exactly
    /// ONE run covering all reachable nodes, single-entry / single-exit.
    #[test]
    fn straight_line_extracts_to_one_run() {
        let mut g = Graph::new();
        let a = f32_node(&mut g, Op::Const, vec![]);
        let b = f32_node(&mut g, Op::Relu, vec![a]);
        let c = f32_node(&mut g, Op::Silu, vec![b]);
        let d = f32_node(&mut g, Op::Tanh, vec![c]);

        let runs = extract_runs(&g, d);
        assert_eq!(runs.len(), 1, "a straight-line graph is exactly one run");
        let run = &runs[0];
        assert_eq!(run.entry, a, "the run enters at the source");
        assert_eq!(run.exit, d, "the run exits at the root");
        assert_eq!(
            run.members,
            vec![a, b, c, d],
            "the single run covers every reachable node in topo order",
        );
        // single-device: no explicit backend anywhere -> None.
        assert_eq!(run.device, None);
        // lower_run reproduces the ordered members.
        assert_eq!(lower_run(run), &[a, b, c, d]);
    }

    /// (b) A finalized 2-arm branch extracts to exactly four runs —
    /// {pre-run, arm0-run, arm1-run, post-run} — each single-entry /
    /// single-exit.
    #[test]
    fn two_arm_branch_extracts_to_four_runs() {
        let (g, diverge, arm0, arm1, reconverge, _branch, post) = diamond_with_branch();

        let runs = extract_runs(&g, post);

        // Each run is single-entry/single-exit by construction.
        for r in &runs {
            assert_eq!(r.entry, r.members[0]);
            assert_eq!(r.exit, *r.members.last().unwrap());
            assert!(!r.members.is_empty());
        }

        // Locate the run that contains each landmark node.
        let run_of = |node: NodeId| -> usize {
            runs.iter()
                .position(|r| r.members.contains(&node))
                .unwrap_or_else(|| panic!("no run contains Node#{}", node.0))
        };
        let pre_run = run_of(diverge);
        let arm0_run = run_of(arm0);
        let arm1_run = run_of(arm1);
        let post_run = run_of(reconverge);

        // Four distinct runs: pre, arm0, arm1, post.
        let distinct: HashSet<usize> =
            [pre_run, arm0_run, arm1_run, post_run].into_iter().collect();
        assert_eq!(
            distinct.len(),
            4,
            "a 2-arm branch yields exactly {{pre, arm0, arm1, post}} = 4 runs; got runs={runs:?}",
        );
        assert_eq!(runs.len(), 4, "no run beyond the four; got {runs:?}");

        // arm0 and arm1 are in *different* runs (a run never spans a
        // branch boundary).
        assert_ne!(arm0_run, arm1_run, "the two arms must be separate runs");
        // diverge is upstream of both arms and in neither arm's run.
        assert_ne!(pre_run, arm0_run);
        assert_ne!(pre_run, arm1_run);
        // the post region (reconverge + Branch + post) is its own run.
        assert!(runs[post_run].members.contains(&post));
    }

    /// PR-A4 arm-0 single-route lowering: [`lower_runs_arm0`] follows
    /// arm 0 through the branch (pre, arm0, post) and SKIPS arm 1's run.
    /// arm 1's interior node never appears; arm 0's does. On a branchless
    /// graph it is identical to concatenating [`lower_run`] over the runs.
    #[test]
    fn lower_runs_arm0_follows_arm0_and_skips_other_arms() {
        let (g, diverge, arm0, arm1, reconverge, _branch, post) = diamond_with_branch();

        let order = lower_runs_arm0(&g, &[post]);

        // arm 1's interior is skipped; arm 0 + pre + post execute.
        assert!(
            !order.contains(&arm1),
            "arm-0 lowering skips arm 1's run; order={order:?} arm1={arm1:?}",
        );
        assert!(order.contains(&diverge), "the pre-run (diverge) executes");
        assert!(order.contains(&arm0), "arm 0 executes");
        assert!(order.contains(&reconverge), "the reconverge executes");
        assert!(order.contains(&post), "the post-run executes");
        // The phi/merge Branch node is structural — never an executable
        // member.
        for &id in &order {
            assert!(
                !matches!(g.node(id).op, Op::Branch { .. }),
                "the Branch node is never an executable member",
            );
        }

        // Branchless: identical to concatenating lower_run over runs.
        let mut g2 = Graph::new();
        let a = f32_node(&mut g2, Op::Const, vec![]);
        let b = f32_node(&mut g2, Op::Relu, vec![a]);
        let c = f32_node(&mut g2, Op::Silu, vec![b]);
        let d = f32_node(&mut g2, Op::Tanh, vec![c]);
        let flat: Vec<NodeId> = extract_runs(&g2, d)
            .iter()
            .flat_map(|r| lower_run(r).to_vec())
            .collect();
        assert_eq!(
            lower_runs_arm0(&g2, &[d]),
            flat,
            "on a branchless graph the arm-0 lowering equals the flat concatenation",
        );
    }

    /// PR-C1 (e) route-aware lowering: [`lower_picked_route`] follows the
    /// **chosen** arm and skips the non-chosen arms' runs, and equals
    /// [`lower_runs_arm0`] when every pick is arm 0 (the empty route).
    ///
    /// On the 2-arm diamond:
    /// - empty route (== arm 0 everywhere): arm 0 in, arm 1 out, and
    ///   byte-identical to `lower_runs_arm0`.
    /// - pick arm 1 for the branch: arm 1 in, arm 0's interior out — the
    ///   mirror image, proving the lowering follows the picked arm rather
    ///   than hard-coding arm 0.
    #[test]
    fn lower_picked_route_follows_chosen_arm_and_skips_others() {
        let (g, _diverge, arm0, arm1, _reconverge, branch, post) =
            diamond_with_branch();

        // (1) Empty route == arm-0 everywhere == lower_runs_arm0.
        let empty = PickedRoute::new();
        let route_order = lower_picked_route(&g, &[post], &empty);
        let arm0_order = lower_runs_arm0(&g, &[post]);
        assert_eq!(
            route_order, arm0_order,
            "the empty route lowers byte-identically to lower_runs_arm0",
        );
        assert!(route_order.contains(&arm0), "empty route follows arm 0");
        assert!(
            !route_order.contains(&arm1),
            "empty route skips arm 1's run; order={route_order:?}",
        );

        // (2) Pick arm 1 for the branch — now arm 1 is followed, arm 0's
        // interior is skipped.
        let mut picked = PickedRoute::new();
        picked.insert(branch, 1);
        let order = lower_picked_route(&g, &[post], &picked);
        assert!(
            order.contains(&arm1),
            "picking arm 1 follows arm 1's run; order={order:?}",
        );
        assert!(
            !order.contains(&arm0),
            "picking arm 1 skips arm 0's interior run; order={order:?}",
        );
        // The Branch node is never an executable member regardless of pick.
        for &id in &order {
            assert!(
                !matches!(g.node(id).op, Op::Branch { .. }),
                "the Branch node is never an executable member",
            );
        }

        // (3) A branch absent from the route still defaults to arm 0.
        let absent = PickedRoute::new();
        let absent_order = lower_picked_route(&g, &[post], &absent);
        assert_eq!(
            absent_order, arm0_order,
            "a branch the picker did not resolve defaults to arm 0",
        );

        // (4) An out-of-range pick degrades to arm 0 (never panic).
        let mut bad = PickedRoute::new();
        bad.insert(branch, 99);
        let bad_order = lower_picked_route(&g, &[post], &bad);
        assert_eq!(
            bad_order, arm0_order,
            "an out-of-range arm index clamps to arm 0 rather than panicking",
        );
    }

    /// Build two coupled diamonds back-to-back (diamond-2 diverges off
    /// diamond-1's reconverge) so the streaming-walk equality test exercises
    /// upstream-first lazy resolution over more than one branch. Returns the
    /// graph + `(branch1, branch2, post, arm1_of_b1, arm1_of_b2)`.
    fn two_coupled_diamonds() -> (Graph, NodeId, NodeId, NodeId, NodeId, NodeId) {
        let mut g = Graph::new();
        let pre = f32_node(&mut g, Op::Const, vec![]);
        // diamond 1
        let div1 = f32_node(&mut g, Op::Relu, vec![pre]);
        let a0_1 = f32_node(&mut g, Op::Silu, vec![div1]);
        let a1_1 = f32_node(&mut g, Op::Gelu, vec![div1]);
        let recon1 = f32_node(&mut g, Op::Relu, vec![a0_1]);
        let mut b1 = g.open_branch(div1);
        b1.add_arm(a0_1);
        b1.add_arm(a1_1);
        let branch1 = b1
            .finalize_branches(&mut g, recon1)
            .expect("branch1 valid")
            .expect("2 arms");
        // diamond 2 (downstream of diamond 1's merge)
        let div2 = f32_node(&mut g, Op::Tanh, vec![recon1]);
        let a0_2 = f32_node(&mut g, Op::Silu, vec![div2]);
        let a1_2 = f32_node(&mut g, Op::Gelu, vec![div2]);
        let recon2 = f32_node(&mut g, Op::Relu, vec![a0_2]);
        let mut b2 = g.open_branch(div2);
        b2.add_arm(a0_2);
        b2.add_arm(a1_2);
        let branch2 = b2
            .finalize_branches(&mut g, recon2)
            .expect("branch2 valid")
            .expect("2 arms");
        let post = f32_node(&mut g, Op::Tanh, vec![recon2]);
        (g, branch1, branch2, post, a1_1, a1_2)
    }

    /// **C1 byte-identity gate (fuel-graph layer).** The streaming lowering
    /// — which resolves each branch lazily, at the run-walk frontier, via a
    /// closure — emits the SAME `NodeId` order as the one-shot
    /// [`lower_picked_route`] over the route the same closure would produce,
    /// for every combination of arm picks across one and two branches. This
    /// is the structural proof that C1 changes *when* a branch resolves, not
    /// *which* arm it picks (or the order it emits).
    #[test]
    fn streaming_lowering_equals_one_shot_on_every_pick() {
        // --- single diamond: all four pick combinations ---
        let (g, _diverge, _arm0, _arm1, _reconverge, branch, post) =
            diamond_with_branch();
        for chosen in [0usize, 1, 99] {
            // One-shot: build the route then lower.
            let mut route = PickedRoute::new();
            if chosen != 0 {
                route.insert(branch, chosen);
            }
            let one_shot = lower_picked_route(&g, &[post], &route);
            // Streaming: resolve the branch lazily at its arm-entry.
            let streamed = lower_picked_route_streaming(&g, &[post], |b| {
                assert_eq!(b, branch, "only the diamond's branch is resolved");
                chosen
            });
            assert_eq!(
                streamed, one_shot,
                "streamed order must equal one-shot for branch pick {chosen}",
            );
        }

        // --- two coupled diamonds: every (arm1, arm2) combination ---
        let (g2, branch1, branch2, post2, _a1_1, _a1_2) = two_coupled_diamonds();
        for c1 in [0usize, 1] {
            for c2 in [0usize, 1] {
                let mut route = PickedRoute::new();
                if c1 != 0 {
                    route.insert(branch1, c1);
                }
                if c2 != 0 {
                    route.insert(branch2, c2);
                }
                let one_shot = lower_picked_route(&g2, &[post2], &route);

                // The streaming closure resolves each branch at most once
                // and upstream-first (branch1 before branch2). Assert that
                // discipline AND the resulting byte-identity.
                let mut seen: Vec<NodeId> = Vec::new();
                let streamed = lower_picked_route_streaming(&g2, &[post2], |b| {
                    assert!(
                        !seen.contains(&b),
                        "each branch is resolved at most once; b={b:?} seen={seen:?}",
                    );
                    seen.push(b);
                    if b == branch1 {
                        c1
                    } else if b == branch2 {
                        c2
                    } else {
                        panic!("unexpected branch {b:?}")
                    }
                });
                assert_eq!(
                    seen,
                    vec![branch1, branch2],
                    "coupled branches resolve upstream-first at the frontier",
                );
                assert_eq!(
                    streamed, one_shot,
                    "streamed == one-shot for picks ({c1},{c2})",
                );
            }
        }
    }

    /// (c) A `target_backend` change mid-chain (set via the existing
    /// accessor) starts a new run — a residency seam.
    #[test]
    fn residency_seam_starts_new_run() {
        let mut g = Graph::new();
        let a = f32_node(&mut g, Op::Const, vec![]);
        let b = f32_node(&mut g, Op::Relu, vec![a]);
        let c = f32_node(&mut g, Op::Silu, vec![b]);
        let d = f32_node(&mut g, Op::Tanh, vec![c]);
        // a,b live on CPU; c,d live on CUDA -> a seam between b and c.
        g.set_target_backend(a, BackendId::Cpu);
        g.set_target_backend(b, BackendId::Cpu);
        g.set_target_backend(c, BackendId::Cuda);
        g.set_target_backend(d, BackendId::Cuda);

        let runs = extract_runs(&g, d);
        assert_eq!(
            runs.len(),
            2,
            "a residency change mid-chain cuts the chain into two runs; got {runs:?}",
        );
        assert_eq!(runs[0].members, vec![a, b]);
        assert_eq!(runs[0].device, Some(BackendId::Cpu));
        assert_eq!(runs[1].members, vec![c, d]);
        assert_eq!(runs[1].device, Some(BackendId::Cuda));
    }

    /// (d) A multi-predecessor fan-in starts a fresh run: the joining
    /// node is the entry of a new run, distinct from either feeder run.
    #[test]
    fn fan_in_starts_new_run() {
        let mut g = Graph::new();
        // Two independent feeders that join at `sum`.
        let a = f32_node(&mut g, Op::Const, vec![]);
        let a1 = f32_node(&mut g, Op::Relu, vec![a]);
        let b = f32_node(&mut g, Op::Const, vec![]);
        let b1 = f32_node(&mut g, Op::Silu, vec![b]);
        let sum = f32_node(&mut g, Op::Add, vec![a1, b1]);

        let runs = extract_runs(&g, sum);
        // The fan-in node `sum` must be the entry of its own run.
        let sum_run = runs
            .iter()
            .find(|r| r.members.contains(&sum))
            .expect("sum is in some run");
        assert_eq!(
            sum_run.entry, sum,
            "a fan-in node starts a fresh run (it is that run's entry)",
        );
        // `sum` shares a run with neither feeder chain.
        assert!(
            !sum_run.members.contains(&a1) && !sum_run.members.contains(&b1),
            "the fan-in run does not absorb either feeder; got {sum_run:?}",
        );
    }

    /// (e) Fewness gate — PASS side: a graph that branches only at a
    /// "layer" boundary sits below the ~5% threshold.
    #[test]
    fn fewness_gate_passes_on_sparse_branching() {
        // A long straight-line "model" with a single deliberate branch.
        let mut g = Graph::new();
        let mut prev = f32_node(&mut g, Op::Const, vec![]);
        // ~40 ops of straight-line body.
        let mut body: Vec<NodeId> = vec![prev];
        for _ in 0..40 {
            prev = f32_node(&mut g, Op::Relu, vec![prev]);
            body.push(prev);
        }
        // One deliberate fork off `prev` (the layer boundary).
        let diverge = prev;
        let arm0 = f32_node(&mut g, Op::Silu, vec![diverge]);
        let arm1 = f32_node(&mut g, Op::Gelu, vec![diverge]);
        let reconverge = f32_node(&mut g, Op::Relu, vec![arm0]);
        let mut b = g.open_branch(diverge);
        b.add_arm(arm0);
        b.add_arm(arm1);
        b.finalize_branches(&mut g, reconverge)
            .expect("valid branch")
            .expect("2 arms survive");
        let root = f32_node(&mut g, Op::Tanh, vec![reconverge]);

        let density = branch_density(&g, root);
        assert!(
            density < FEWNESS_THRESHOLD,
            "sparse layer-boundary branching must pass the fewness gate; density={density}",
        );
        assert!(passes_fewness_gate(&g, root));
    }

    /// (e) Fewness gate — FAIL side: a synthetic graph that forks at
    /// (nearly) every op blows past the ~5% threshold.
    #[test]
    fn fewness_gate_fails_on_per_op_branching() {
        // Build a chain of N diamonds back-to-back, so a large fraction
        // of nodes are Op::Branch decision points.
        let mut g = Graph::new();
        let mut prev = f32_node(&mut g, Op::Const, vec![]);
        for _ in 0..12 {
            let diverge = prev;
            let arm0 = f32_node(&mut g, Op::Silu, vec![diverge]);
            let arm1 = f32_node(&mut g, Op::Gelu, vec![diverge]);
            let reconverge = f32_node(&mut g, Op::Relu, vec![arm0]);
            let mut b = g.open_branch(diverge);
            b.add_arm(arm0);
            b.add_arm(arm1);
            b.finalize_branches(&mut g, reconverge)
                .expect("valid branch")
                .expect("2 arms survive");
            prev = reconverge;
        }
        let root = prev;

        let density = branch_density(&g, root);
        assert!(
            density >= FEWNESS_THRESHOLD,
            "per-op branching must FAIL the fewness gate; density={density}",
        );
        assert!(!passes_fewness_gate(&g, root));
    }

    // ===== PR-C3: device-alternating reorder (auto-overlap) =====

    /// Build the "arbitrary graph" the auto-overlap benchmark needs WITHOUT
    /// the hand-constructed crutch: two INDEPENDENT same-length device
    /// chains (one CUDA, one Vulkan) over distinct consts, reconverging at a
    /// final CUDA `add`. The reconverge inputs are `[cuda_root, vulkan_root]`
    /// — the "wrong" order that the un-reordered topo DFS pops CUDA-last, so
    /// it would emit the Vulkan chain + its cross-device copy FIRST (host
    /// blocks on Vulkan before CUDA is enqueued → serialized). One explicit
    /// `Op::Copy{target:Cuda}` bridges the Vulkan root to the CUDA
    /// reconverge (a run boundary). Returns `(graph, root, cuda_chain_nodes,
    /// vk_chain_nodes, copy_node)`.
    fn two_device_reconverge(
        cuda_len: usize,
        vk_len: usize,
    ) -> (Graph, NodeId, Vec<NodeId>, Vec<NodeId>, NodeId) {
        let mut g = Graph::new();
        // Distinct consts, one per device (CPU-resident sources H2D'd to
        // each device — modeled here as the chain heads carrying the device
        // stamp; the const itself is unplaced/default).
        let cc = f32_node(&mut g, Op::Const, vec![]);
        let vc = f32_node(&mut g, Op::Const, vec![]);

        // CUDA chain.
        let mut cuda_nodes = Vec::new();
        let mut cprev = cc;
        for _ in 0..cuda_len {
            let n = f32_node(&mut g, Op::Relu, vec![cprev]);
            g.set_target_backend(n, BackendId::Cuda);
            cuda_nodes.push(n);
            cprev = n;
        }
        // Vulkan chain.
        let mut vk_nodes = Vec::new();
        let mut vprev = vc;
        for _ in 0..vk_len {
            let n = f32_node(&mut g, Op::Silu, vec![vprev]);
            g.set_target_backend(n, BackendId::Vulkan);
            vk_nodes.push(n);
            vprev = n;
        }
        // Cross-device copy: Vulkan root -> CUDA (a residency seam = its own
        // run). Reconverge on CUDA reads [cuda_root, copy].
        let copy = g.push(Node {
            op: Op::Copy { target: fuel_ir::DeviceLocation::Cuda { gpu_id: 0 } },
            inputs: vec![vprev],
            shape: Shape::from_dims(&[2]),
            dtype: DType::F32,
        });
        g.set_target_backend(copy, BackendId::Cuda);
        // Reconverge inputs in the "wrong" (cuda-first) order, so the DFS
        // pops the LAST input (the Vulkan-fed copy) first.
        let out = g.push(Node {
            op: Op::Add,
            inputs: vec![cprev, copy],
            shape: Shape::from_dims(&[2]),
            dtype: DType::F32,
        });
        g.set_target_backend(out, BackendId::Cuda);
        (g, out, cuda_nodes, vk_nodes, copy)
    }

    /// Assert `order` is a valid topological reordering of `runs`: every run
    /// appears after all the runs it depends on (a member reads a member of
    /// the producer run).
    fn assert_valid_topo_run_order(graph: &Graph, runs: &[Run], order: &[usize]) {
        let mut run_of: HashMap<NodeId, usize> = HashMap::new();
        for (ri, r) in runs.iter().enumerate() {
            for &m in &r.members {
                run_of.insert(m, ri);
            }
        }
        let mut emitted: HashSet<usize> = HashSet::new();
        for &ri in order {
            for &m in &runs[ri].members {
                for &inp in &graph.node(m).inputs {
                    if let Some(&pi) = run_of.get(&inp) {
                        if pi != ri {
                            assert!(
                                emitted.contains(&pi),
                                "run {ri} emitted before its producer run {pi}",
                            );
                        }
                    }
                }
            }
            emitted.insert(ri);
        }
        assert_eq!(order.len(), runs.len(), "every run emitted exactly once");
        let distinct: HashSet<usize> = order.iter().copied().collect();
        assert_eq!(distinct.len(), runs.len(), "the order is a permutation");
    }

    /// C3 core: a two-device run list is reordered into a valid topological
    /// order that INTERLEAVES the devices — the independent CUDA and Vulkan
    /// producer chunks are both emitted BEFORE the cross-device copy that
    /// feeds the reconverge (the A4b overlap topology), regardless of the
    /// reconverge's input order.
    #[test]
    fn device_alternating_reorder_interleaves_two_devices() {
        let (g, out, cuda_nodes, vk_nodes, copy) = two_device_reconverge(4, 4);
        let runs = extract_runs(&g, out);
        let perm = device_alternating_order(&g, &runs);

        // (1) A valid topological reordering.
        assert_valid_topo_run_order(&g, &runs, &perm);

        // Map each landmark node to its position in the FLATTENED reordered
        // order, so we can assert the overlap-relevant ordering.
        let flat: Vec<NodeId> =
            perm.iter().flat_map(|&ri| runs[ri].members.clone()).collect();
        let pos = |n: NodeId| flat.iter().position(|&x| x == n).expect("node present");

        let cuda_root = *cuda_nodes.last().unwrap();
        let vk_root = *vk_nodes.last().unwrap();

        // (2) BOTH device producer chunks are emitted before the
        //     cross-device copy (the host-blocking D2H). This is the
        //     property the un-reordered cuda-last DFS violates and that the
        //     A4b eager-submit mechanism turns into overlap.
        assert!(
            pos(cuda_root) < pos(copy),
            "the CUDA chunk must be enqueued before the cross-device copy; \
             flat={flat:?}",
        );
        assert!(
            pos(vk_root) < pos(copy),
            "the Vulkan chunk must be recorded before the cross-device copy; \
             flat={flat:?}",
        );

        // (3) The devices INTERLEAVE rather than one draining fully first:
        //     the run-device sequence has at least one CUDA<->Vulkan
        //     transition before the copy. (With one chunk per device this is
        //     just "both chunks precede the copy", already asserted; assert
        //     the device sequence explicitly too.)
        let dev_seq: Vec<Option<BackendId>> =
            perm.iter().map(|&ri| runs[ri].device).collect();
        let switches = dev_seq
            .windows(2)
            .filter(|w| w[0] != w[1])
            .count();
        assert!(
            switches >= 1,
            "a two-device run list must contain at least one device switch; \
             dev_seq={dev_seq:?}",
        );
    }

    /// C3 gate: a SINGLE-device run list is the IDENTITY permutation — the
    /// reorder never diverges from the input order, so the lowering is
    /// byte-identical to today's (the primary single-device regression
    /// gate).
    #[test]
    fn device_alternating_reorder_single_device_is_identity() {
        // All-CUDA chain (one distinct device) → identity.
        let mut g = Graph::new();
        let mut prev = f32_node(&mut g, Op::Const, vec![]);
        for _ in 0..6 {
            prev = f32_node(&mut g, Op::Relu, vec![prev]);
            g.set_target_backend(prev, BackendId::Cuda);
        }
        let runs = extract_runs(&g, prev);
        let perm = device_alternating_order(&g, &runs);
        assert_eq!(
            perm,
            (0..runs.len()).collect::<Vec<_>>(),
            "a single-device run list reorders to the identity permutation",
        );

        // A graph with NO explicit backend anywhere (all device == None) is
        // also a single bucket → identity (the branchless single-route CPU
        // graph — must stay byte-identical).
        let mut g2 = Graph::new();
        let a = f32_node(&mut g2, Op::Const, vec![]);
        let b = f32_node(&mut g2, Op::Relu, vec![a]);
        let c = f32_node(&mut g2, Op::Silu, vec![b]);
        let runs2 = extract_runs(&g2, c);
        assert_eq!(
            device_alternating_order(&g2, &runs2),
            (0..runs2.len()).collect::<Vec<_>>(),
            "an all-default (device=None) run list is identity",
        );
    }

    /// C3 correctness: the reorder NEVER violates a data dependency, even
    /// when alternation "wants" to move a run earlier. A linear cross-device
    /// chain CPU→CUDA→Vulkan→CUDA (each a residency seam, so 4 runs, each
    /// depending strictly on the prior) must come out in dependency order —
    /// the device preference can't reorder a strict chain.
    #[test]
    fn device_alternating_reorder_respects_dependencies() {
        let mut g = Graph::new();
        let a = f32_node(&mut g, Op::Const, vec![]);
        g.set_target_backend(a, BackendId::Cpu);
        let b = f32_node(&mut g, Op::Relu, vec![a]);
        g.set_target_backend(b, BackendId::Cuda);
        let c = f32_node(&mut g, Op::Silu, vec![b]);
        g.set_target_backend(c, BackendId::Vulkan);
        let d = f32_node(&mut g, Op::Tanh, vec![c]);
        g.set_target_backend(d, BackendId::Cuda);

        let runs = extract_runs(&g, d);
        let perm = device_alternating_order(&g, &runs);
        assert_valid_topo_run_order(&g, &runs, &perm);
        // A strict chain has exactly one valid topological order; the
        // alternation cannot change it.
        let flat: Vec<NodeId> =
            perm.iter().flat_map(|&ri| runs[ri].members.clone()).collect();
        assert_eq!(
            flat,
            vec![a, b, c, d],
            "a strict cross-device chain keeps dependency order despite \
             device alternation",
        );
    }

    /// C3 invariance (fuel-graph layer): the device-alternating lowering of
    /// a multi-device graph emits the SAME multiset of NodeIds as the
    /// un-reordered concatenation, in a VALID topological order — only the
    /// order across independent runs changes. (The byte-identical-OUTPUT
    /// proof is the live reorder-invariance test in `fuel-dispatch`; this is
    /// the structural guard that the reorder is a permutation, not a
    /// rewrite.)
    #[test]
    fn lower_picked_route_multidevice_is_a_valid_reordering() {
        let (g, out, _cuda, _vk, _copy) = two_device_reconverge(4, 4);
        let runs = extract_runs(&g, out);

        // Un-reordered reference: concatenate runs in topo order.
        let reference: Vec<NodeId> =
            runs.iter().flat_map(|r| lower_run(r).to_vec()).collect();
        // The production lowering (now reorders multi-device).
        let lowered = lower_picked_route(&g, &[out], &PickedRoute::new());

        // Same multiset of nodes (a permutation, no add/drop).
        let mut a = reference.clone();
        let mut b = lowered.clone();
        a.sort();
        b.sort();
        assert_eq!(a, b, "the reorder is a permutation of the same node set");

        // The lowered order is a valid topological order of the graph: every
        // node appears after all of its (reachable, lowered) inputs.
        let lowered_pos: HashMap<NodeId, usize> =
            lowered.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        for (i, &n) in lowered.iter().enumerate() {
            for &inp in &g.node(n).inputs {
                if let Some(&j) = lowered_pos.get(&inp) {
                    assert!(j < i, "input {inp:?} of {n:?} must precede it");
                }
            }
        }

        // And it genuinely diverged from the reference (the reorder fired) —
        // the cuda-last DFS reference puts the Vulkan-fed copy's chain before
        // the CUDA chain; the reorder interleaves them.
        assert_ne!(
            lowered, reference,
            "on this two-device graph the reorder must change the order \
             (else auto-overlap did nothing)",
        );
    }

    /// C3 + C1 composition: the STREAMING lowering of a multi-device graph
    /// equals the one-shot `lower_picked_route` (both now reorder), and the
    /// streaming reorder is confined to branch-free segments. On a
    /// branchless multi-device graph there is exactly one segment, so the
    /// streamed order equals the one-shot reordered order byte-for-byte.
    #[test]
    fn streaming_multidevice_equals_one_shot_reordered() {
        let (g, out, _cuda, _vk, _copy) = two_device_reconverge(4, 4);
        let one_shot = lower_picked_route(&g, &[out], &PickedRoute::new());
        let streamed = lower_picked_route_streaming(&g, &[out], |_b| 0);
        assert_eq!(
            streamed, one_shot,
            "streamed multi-device order equals the one-shot reordered order",
        );
    }

    /// C3 critical-path property (the live-overlap regression guard): when
    /// the two device chunks differ in size, the reorder emits the HEAVIER
    /// chunk's whole path FIRST — so the auto-submitting device (the one with
    /// the long chain) is enqueued and running before the lighter device's
    /// host-blocking drain. This mirrors the live benchmark (CUDA-768 vs
    /// Vulkan-120): the proven-overlap order is CUDA-chunk-first; the inverse
    /// (Vulkan-first) serialized at 0.0 efficiency. This test locks the
    /// heavy-first ordering that makes the arbitrary-graph benchmark
    /// auto-overlap.
    #[test]
    fn device_alternating_reorder_emits_heavier_chunk_first() {
        // CUDA chain HEAVY (8), Vulkan chain LIGHT (2). Reconverge inputs are
        // cuda-first (the "wrong" DFS order). The reorder must still emit the
        // heavy CUDA chunk's path before the light Vulkan chunk.
        let (g, out, cuda_nodes, vk_nodes, copy) = two_device_reconverge(8, 2);
        let runs = extract_runs(&g, out);
        let perm = device_alternating_order(&g, &runs);
        assert_valid_topo_run_order(&g, &runs, &perm);

        let flat: Vec<NodeId> =
            perm.iter().flat_map(|&ri| runs[ri].members.clone()).collect();
        let pos = |n: NodeId| flat.iter().position(|&x| x == n).expect("present");
        let cuda_root = *cuda_nodes.last().unwrap();
        let vk_root = *vk_nodes.last().unwrap();

        // The HEAVY (CUDA) chunk's root is emitted BEFORE the LIGHT (Vulkan)
        // chunk's root — the heavy device is enqueued first.
        assert!(
            pos(cuda_root) < pos(vk_root),
            "the heavier CUDA chunk must be emitted before the lighter Vulkan \
             chunk (the auto-submit device runs first); flat={flat:?}",
        );
        // And BOTH chunks precede the host-blocking cross-device copy.
        assert!(pos(cuda_root) < pos(copy), "heavy chunk before the drain");
        assert!(pos(vk_root) < pos(copy), "light chunk before the drain");

        // Symmetric check: make VULKAN the heavy chunk; now Vulkan's path
        // must come first. (Proves it's size-driven, not a CUDA hard-code.)
        let (g2, out2, cuda2, vk2, copy2) = two_device_reconverge(2, 8);
        let runs2 = extract_runs(&g2, out2);
        let perm2 = device_alternating_order(&g2, &runs2);
        assert_valid_topo_run_order(&g2, &runs2, &perm2);
        let flat2: Vec<NodeId> =
            perm2.iter().flat_map(|&ri| runs2[ri].members.clone()).collect();
        let pos2 = |n: NodeId| flat2.iter().position(|&x| x == n).expect("present");
        assert!(
            pos2(*vk2.last().unwrap()) < pos2(*cuda2.last().unwrap()),
            "when Vulkan is the heavier chunk it is emitted first; flat2={flat2:?}",
        );
        assert!(pos2(*vk2.last().unwrap()) < pos2(copy2));
        assert!(pos2(*cuda2.last().unwrap()) < pos2(copy2));
    }
}

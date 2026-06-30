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
    let mut order = Vec::new();
    for run in &runs {
        // A run is a single contiguous arm/region (extract never spans a
        // branch boundary), so it is either wholly inside a non-chosen
        // arm or wholly outside. Skip iff its entry is a skipped node.
        if skip.contains(&run.entry) {
            continue;
        }
        order.extend_from_slice(lower_run(run));
    }
    order
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
}

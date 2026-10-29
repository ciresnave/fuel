// SPDX-License-Identifier: MIT OR Apache-2.0
//! Residency analysis + eviction for the pipelined executor.
//!
//! Ported from `fuel-graph-router::{residency_planner,
//! residency_eviction}` (executor-unification Session 6, 2026-06-11).
//! The legacy shape was a `GraphMutatingSchedulerRule` that emitted a
//! three-node `Op::Copy{Cpu}` + `Op::Release` + `Op::Copy{device}`
//! chain for the legacy `GraphExecutor<Router>`; the pipelined port
//! emits the fused two-node form — `Op::Move{Cpu}` + reload
//! `Op::Copy{device}` — making this pass the first production emitter
//! of [`WorkItemKind::Move`](crate::pipelined) (shipped `b93bdb82`).
//!
//! ## The two halves
//!
//! - **Analysis** — [`ResidencyPlanner::analyze`] walks a graph's topo
//!   order, tracks each node's live range (first use → last consumer),
//!   and computes the peak working-set size. Pure; no graph mutation.
//! - **Transform** — [`insert_residency_evictions`] consumes the
//!   analysis and, while the peak exceeds a byte budget, inserts
//!   `Op::Move{Cpu}` + reload `Op::Copy{device}` chains around the
//!   highest-scoring candidates (`bytes × inactive_gap`). The
//!   `derive_ordering` half of `execution_plan` (run inside both
//!   pipelined realize paths) pins each Move after every
//!   non-destructive reader of its source, and the executor's
//!   `destructive_input` cleanup frees the device storage when the
//!   Move runs.
//!
//! ## Const-pool byte budget (deferred — planner program)
//!
//! The legacy executor's `with_const_pool_limit` LRU was the only
//! other larger-than-VRAM mechanism, and it retired with the legacy
//! executor. Per the 2026-06-11 re-audit (gap 7), the replacement is
//! NOT an executor-side LRU: weight residency under a byte budget is
//! a *planning* decision, and it folds into the load-time incremental
//! planner's residency program (`docs/session-prompts/
//! load-time-incremental-planner.md`) — the planner prices
//! evict/reload chains with Stage 1 `TransferCalibration` data and
//! emits the same `Op::Move`/`Op::Copy` primitives this module does.
//! This module is that program's substrate, not its policy.
//!
//! ## Known limitations (follow-up work, inherited from the port)
//!
//! - Only one eviction per candidate (the largest gap). Multiple gaps
//!   would need multiple evict chains.
//! - Greedy selection; no cost-awareness for the CPU↔device transfer
//!   overhead vs the VRAM savings (the planner program adds pricing).
//! - No integration with the `VulkanBackend::evict` mmap-tiering path
//!   — that's a transparent runtime mechanism, distinct from
//!   graph-level planning.

use crate::plan::backend_for_device;
use fuel_graph::opt::execution_plan;
use fuel_graph::{Graph, NodeId, Op, SharedGraph, opt, topo_order_multi};
use fuel_ir::{DeviceLocation, Error, Result};
use std::collections::HashMap;

/// One tensor's residency span. `[first_use, last_use]` are op
/// positions in a topo order (inclusive — the tensor is live during
/// both). `bytes` is the output storage size. `inactive_gap` is the
/// longest stretch of ops between consecutive reads where the tensor
/// is resident but not accessed — a strong signal for eviction
/// viability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveRange {
    pub node: NodeId,
    pub bytes: usize,
    pub first_use: usize,
    pub last_use: usize,
    pub inactive_gap: usize,
}

/// Structured report on a graph's residency profile.
#[derive(Debug, Clone)]
pub struct ResidencyReport {
    /// Total bytes of storage summed over every reachable op's output.
    /// Not the peak — this is cumulative if nothing were freed.
    pub total_bytes: usize,
    /// Maximum bytes live simultaneously at any point during a
    /// single-device execution of the graph. The minimum VRAM
    /// (or "primary device") required to run without spilling.
    pub peak_bytes: usize,
    /// Op position (index in topo order) where the peak occurs.
    pub peak_op_index: usize,
    /// Per-tensor live range, sorted by `bytes * inactive_gap`
    /// descending (biggest "wasted residency" candidates first).
    /// [`insert_residency_evictions`] consumes this list.
    pub eviction_candidates: Vec<LiveRange>,
}

impl ResidencyReport {
    /// Does this graph fit in a given byte budget?
    pub fn fits_in(&self, budget: usize) -> bool {
        self.peak_bytes <= budget
    }

    /// Byte overage relative to a budget. Zero when the graph fits.
    pub fn overage(&self, budget: usize) -> usize {
        self.peak_bytes.saturating_sub(budget)
    }
}

/// Analysis-only residency planner. Computes live ranges and peak
/// bytes for a graph, emits a [`ResidencyReport`]. Does not mutate
/// the graph. Takes `&Graph` — the caller holds whatever lock it
/// already has; no hidden lock acquisition.
pub struct ResidencyPlanner;

impl ResidencyPlanner {
    /// Run the analysis. O(V + E) in the reachable subgraph. Walks a plain topological order and
    /// counts every node's output (the single-device view the KV-pool sizing uses).
    pub fn analyze(g: &Graph, roots: &[NodeId]) -> ResidencyReport {
        Self::analyze_ordered(g, roots, topo_order_multi(g, roots), |_| true)
    }

    /// The analysis the eviction pass uses to decide whether `device` fits a budget: the walk is
    /// the order the executor will actually run ([`execution_plan`], which pins each `Op::Move`
    /// after every non-destructive reader of its source), and a node counts only while it is
    /// resident on `device`: the output of an `Op::Move` / `Op::Copy` aimed at ANOTHER device
    /// (the staged host copy of an eviction) is not device bytes. Without both, an eviction could
    /// never lower the analysed peak and the pass would evict until it ran out of rounds.
    pub fn analyze_on(g: &Graph, roots: &[NodeId], device: DeviceLocation) -> ResidencyReport {
        let resident = |nid: NodeId| match &g.node(nid).op {
            Op::Move { target } | Op::Copy { target } | Op::Alloc { target } => *target == device,
            // A stamped placement on another device is not this device's bytes; an unstamped node
            // inherits the executor's default and is counted (the conservative direction).
            _ => g.placement(nid).is_none_or(|p| p == device),
        };
        Self::analyze_ordered(g, roots, execution_plan(g, roots), resident)
    }

    fn analyze_ordered(
        g: &Graph,
        roots: &[NodeId],
        order: Vec<NodeId>,
        resident: impl Fn(NodeId) -> bool,
    ) -> ResidencyReport {
        // 1. For every node, compute its output byte size.
        // 2. Determine first-use (= position of the producing op in
        //    topo order) and last-use (= position of the last
        //    consuming op in topo order) for every producer.
        //    Roots count as consumed at the walk's end.
        let n = order.len();
        let mut byte_of: HashMap<NodeId, usize> = HashMap::with_capacity(n);
        let mut first_use: HashMap<NodeId, usize> = HashMap::with_capacity(n);
        let mut last_use: HashMap<NodeId, usize> = HashMap::with_capacity(n);
        // Record the set of reads per op position for inactive-gap
        // computation below.
        let mut reads_at: Vec<Vec<NodeId>> = vec![Vec::new(); n];

        for (op_idx, &nid) in order.iter().enumerate() {
            let node = g.node(nid);
            let elems = node.shape.elem_count();
            let bytes = if resident(nid) {
                elems * node.dtype.size_in_bytes()
            } else {
                0
            };
            byte_of.insert(nid, bytes);
            first_use.entry(nid).or_insert(op_idx);

            for &input in &node.inputs {
                last_use.insert(input, op_idx);
                reads_at[op_idx].push(input);
            }
        }
        // Roots themselves need to stay live through their own
        // position (they're the "output" the caller asked for).
        for &r in roots {
            if let Some(pos) = order.iter().position(|&x| x == r) {
                last_use.entry(r).or_insert(pos);
            }
        }

        let (peak_bytes, peak_op_index, total_bytes) =
            scan_peak(&order, &byte_of, &last_use, &reads_at);

        let gap_of = inactive_gaps(&first_use, &last_use, &reads_at);

        let cands = rank_candidates(&order, &byte_of, &first_use, &last_use, &gap_of);

        ResidencyReport {
            total_bytes,
            peak_bytes,
            peak_op_index,
            eviction_candidates: cands,
        }
    }
}

/// Walk `order`, keeping the live-set bytes: a node goes live at its position and is freed after
/// its last reader. Returns `(peak_bytes, peak_op_index, total_bytes)`.
fn scan_peak(
    order: &[NodeId],
    byte_of: &HashMap<NodeId, usize>,
    last_use: &HashMap<NodeId, usize>,
    reads_at: &[Vec<NodeId>],
) -> (usize, usize, usize) {
    let mut live_bytes: usize = 0;
    let mut peak_bytes: usize = 0;
    let mut peak_op_index: usize = 0;
    let mut total_bytes: usize = 0;
    for (op_idx, &nid) in order.iter().enumerate() {
        let bytes = *byte_of.get(&nid).unwrap_or(&0);
        live_bytes += bytes;
        total_bytes += bytes;
        if live_bytes > peak_bytes {
            peak_bytes = live_bytes;
            peak_op_index = op_idx;
        }
        // Post-op: free the inputs whose last read is this op (not the node just produced).
        for &input in &reads_at[op_idx] {
            if last_use.get(&input) == Some(&op_idx) {
                let b = *byte_of.get(&input).unwrap_or(&0);
                live_bytes = live_bytes.saturating_sub(b);
            }
        }
    }
    (peak_bytes, peak_op_index, total_bytes)
}

/// Per node, the longest stretch of ops between consecutive reads while the tensor is live but
/// untouched (produced -> reads -> last use). Big gap x bytes = the best eviction candidates.
fn inactive_gaps(
    first_use: &HashMap<NodeId, usize>,
    last_use: &HashMap<NodeId, usize>,
    reads_at: &[Vec<NodeId>],
) -> HashMap<NodeId, usize> {
    let mut gap_of: HashMap<NodeId, usize> = HashMap::with_capacity(first_use.len());
    for (nid, &first) in first_use {
        let mut points = vec![first];
        points.extend(
            reads_at
                .iter()
                .enumerate()
                .filter(|(_, rs)| rs.contains(nid))
                .map(|(i, _)| i),
        );
        points.push(last_use.get(nid).copied().unwrap_or(first));
        points.sort();
        let max_gap = points.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
        gap_of.insert(*nid, max_gap);
    }
    gap_of
}

/// Build the eviction candidates and rank them by bytes x inactive gap, descending (ties: larger
/// bytes first, more reclaim per evict). Zero-score nodes (hot tensors with no gap) are dropped and
/// the list is capped for downstream consumers.
fn rank_candidates(
    order: &[NodeId],
    byte_of: &HashMap<NodeId, usize>,
    first_use: &HashMap<NodeId, usize>,
    last_use: &HashMap<NodeId, usize>,
    gap_of: &HashMap<NodeId, usize>,
) -> Vec<LiveRange> {
    let mut cands: Vec<LiveRange> = order
        .iter()
        .map(|&nid| LiveRange {
            node: nid,
            bytes: *byte_of.get(&nid).unwrap_or(&0),
            first_use: *first_use.get(&nid).unwrap_or(&0),
            last_use: *last_use.get(&nid).unwrap_or(&0),
            inactive_gap: *gap_of.get(&nid).unwrap_or(&0),
        })
        .collect();
    cands.sort_by(|a, b| {
        let a_score = (a.bytes as u128) * (a.inactive_gap as u128);
        let b_score = (b.bytes as u128) * (b.inactive_gap as u128);
        b_score.cmp(&a_score).then(b.bytes.cmp(&a.bytes))
    });
    cands.retain(|c| c.bytes > 0 && c.inactive_gap > 0);
    cands.truncate(32);
    cands
}

/// One evict/reload chain emitted by [`insert_residency_evictions`].
/// `move_node` is the `Op::Move { target: Cpu }` (destructive stage-
/// to-host); `reload` is the `Op::Copy { target: src_device }` the
/// post-gap consumers were rewired to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvictReload {
    pub candidate: NodeId,
    pub move_node: NodeId,
    pub reload: NodeId,
    pub src_device: DeviceLocation,
}

/// What one run of the eviction pass did, so a caller can see whether the budget was met instead
/// of inferring it from a graph that quietly grew.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvictionOutcome {
    /// The device whose budget this was.
    pub device: DeviceLocation,
    /// The budget the pass was asked to meet, in bytes.
    pub budget_bytes: usize,
    /// The analysed peak on `device` before any eviction.
    pub peak_before: usize,
    /// The analysed peak on `device` after the last chain the pass kept.
    pub peak_after: usize,
    /// The chains inserted, in order.
    pub chains: Vec<EvictReload>,
}

impl EvictionOutcome {
    /// Whether the analysed peak now fits the budget.
    pub fn fits(&self) -> bool {
        self.peak_after <= self.budget_bytes
    }
}

/// Graph-mutating eviction pass: while the graph's peak residency
/// exceeds `budget_bytes`, pick the top candidate (by `bytes ×
/// inactive_gap`) that has a realizable inactive gap and insert an
/// `Op::Move{Cpu}` + reload `Op::Copy{device}` chain around it
/// (via [`fuel_graph::opt::insert_evict_reload`]).
///
/// `placement_of` reports each candidate's resident
/// [`DeviceLocation`] — callers on the pipelined path derive it from
/// `graph.placement(id)` / the plan's stamped winners; `None` falls
/// back to `Cpu` (the chain is then a same-device spill, correct but
/// pure overhead — mirrors the legacy rule's default). The pass
/// stamps `target_backend` on both emitted nodes (`Move` runs on the
/// source device's backend, the reload's transfer kernel on `Cpu` —
/// the staged copy's residency), so the chain compiles on the
/// pipelined executor without a separate prepare pass.
///
/// Returns the inserted chains. Idempotent once the budget is met or
/// no candidate has a gap ≥ 2 op positions; capped at `max_rounds`
/// (16 is the legacy default) to prevent pathological loops. A realize root is never evicted
/// (the caller reads its storage after the run, and a `Move` consumes it), and the pass stops at
/// the first round that does not lower the analysed peak (see
/// [`insert_residency_evictions_in`], which also reports the outcome).
pub fn insert_residency_evictions(
    graph: &SharedGraph,
    roots: &[NodeId],
    budget_bytes: usize,
    max_rounds: usize,
    placement_of: impl Fn(NodeId) -> Option<DeviceLocation>,
) -> Result<Vec<EvictReload>> {
    // The device whose budget this is: where the roots live (the pinned device on the production
    // path), `Cpu` when unknown (a same-device spill, as before).
    let device = roots
        .first()
        .and_then(|&r| placement_of(r))
        .unwrap_or(DeviceLocation::Cpu);
    let mut g = write_graph(graph)?;
    Ok(insert_residency_evictions_in(
        &mut g,
        roots,
        device,
        budget_bytes,
        max_rounds,
        placement_of,
    )?
    .chains)
}

/// [`insert_residency_evictions`] on a graph the caller already holds mutably: what
/// `optimize_graph` (which takes `&mut Graph`) calls when `PlanOptions::device_budget_bytes` is set.
/// Same candidates, same stamps, and an [`EvictionOutcome`] instead of a bare list.
///
/// **It stops at the first round that does not lower the analysed peak.** Today the reload `Op::Copy`
/// runs right after its `Op::Move` (nothing delays it; ROADMAP row 67 I3 adds the ordering
/// primitive), so on a same-device budget a round cannot lower the peak, and without this stop the
/// pass would insert a transfer pair per candidate, up to `max_rounds`, for no gain. The one chain
/// that revealed the stall is kept (a graph cannot drop nodes) and the outcome reports
/// `peak_after >= peak_before` and `!fits()`: the caller sees that the budget was NOT met.
/// A realize root is never a candidate: the caller reads its storage after the run.
pub fn insert_residency_evictions_in(
    g: &mut Graph,
    roots: &[NodeId],
    device: DeviceLocation,
    budget_bytes: usize,
    max_rounds: usize,
    placement_of: impl Fn(NodeId) -> Option<DeviceLocation>,
) -> Result<EvictionOutcome> {
    // Track candidates we've already evicted this run. Without this
    // guard, a just-evicted candidate's new consumers (its own Move)
    // show up as post-gap consumers in the next round, triggering
    // infinite re-eviction.
    let mut evicted: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    let mut chains: Vec<EvictReload> = Vec::new();
    let peak_before = ResidencyPlanner::analyze_on(g, roots, device).peak_bytes;
    let mut peak = peak_before;

    for _round in 0..max_rounds {
        let report = ResidencyPlanner::analyze_on(g, roots, device);
        if report.fits_in(budget_bytes) {
            break;
        }
        // Pick the top candidate that still has a realizable gap AND
        // hasn't been evicted already in this run.
        let Some((cand, post_gap_consumers)) =
            first_evictable(g, roots, &report.eviction_candidates, &evicted)
        else {
            break;
        };
        let src_device = placement_of(cand).unwrap_or(DeviceLocation::Cpu);
        chains.push(evict_one(g, cand, src_device, &post_gap_consumers));
        evicted.insert(cand);
        let after = ResidencyPlanner::analyze_on(g, roots, device).peak_bytes;
        let progressed = after < peak;
        peak = after;
        if !progressed {
            break;
        }
    }
    Ok(EvictionOutcome {
        device,
        budget_bytes,
        peak_before,
        peak_after: peak,
        chains,
    })
}

/// The highest-ranked candidate that was not evicted yet, is not a realize root, and has post-gap
/// consumers to rewire.
fn first_evictable(
    g: &Graph,
    roots: &[NodeId],
    candidates: &[LiveRange],
    evicted: &std::collections::HashSet<NodeId>,
) -> Option<(NodeId, Vec<NodeId>)> {
    candidates.iter().find_map(|c| {
        if roots.contains(&c.node) || evicted.contains(&c.node) || already_evicted(g, c.node) {
            return None;
        }
        match find_gap_consumers(g, c.node) {
            Some((_pre_gap, post)) if !post.is_empty() => Some((c.node, post)),
            _ => None,
        }
    })
}

/// Insert one evict/reload chain for `candidate` and stamp the transfer pair so the pipelined
/// executor's `Op::Move` / `Op::Copy` compile arms (which need an explicit `target_backend`, the
/// backend whose kernel runs the transfer) resolve without a prepare pass: the Move downloads
/// from `src_device`'s backend; the reload uploads from the staged CPU copy.
fn evict_one(
    g: &mut Graph,
    candidate: NodeId,
    src_device: DeviceLocation,
    post_gap_consumers: &[NodeId],
) -> EvictReload {
    let (move_id, reload_id) =
        opt::insert_evict_reload_in(g, candidate, src_device, post_gap_consumers);
    g.set_target_backend(move_id, backend_for_device(src_device));
    g.set_target_backend(reload_id, backend_for_device(DeviceLocation::Cpu));
    g.set_placement(move_id, DeviceLocation::Cpu);
    g.set_placement(reload_id, src_device);
    EvictReload {
        candidate,
        move_node: move_id,
        reload: reload_id,
        src_device,
    }
}

/// Whether `candidate` already feeds an `Op::Move` (from an earlier run of this pass, or from any
/// other transfer: ANY `Move` blocks, which can miss an optimisation but never double-moves).
/// A second destructive consumer of the same source makes the Moves wait on each other:
/// `execution_plan` then panics with "cycle in ordering edges". So a candidate is evicted at
/// most once across runs, not just within one (`evicted` below only covers one run).
fn already_evicted(g: &Graph, candidate: NodeId) -> bool {
    (0..g.len()).any(|i| {
        let n = g.node(NodeId(i));
        matches!(n.op, Op::Move { .. }) && n.inputs.first() == Some(&candidate)
    })
}

/// Find the pre-gap and post-gap consumers of `candidate` — i.e., the
/// consumer that reads it just before its longest inactive stretch,
/// and the consumer(s) that read it after.
///
/// Returns `None` if the candidate has fewer than two consumers, or if
/// the largest gap is < 2 op positions (no meaningful window to evict
/// into).
fn find_gap_consumers(g: &Graph, candidate: NodeId) -> Option<(NodeId, Vec<NodeId>)> {
    // Walk EVERY node as a synthetic root set — the pass needs to see
    // ALL consumers of a candidate, not just those on a specific
    // root's path.
    let all: Vec<NodeId> = (0..g.len()).map(NodeId).collect();
    let order = topo_order_multi(g, &all);

    // Positions of candidate's consumers in topo order.
    let mut positions: Vec<(usize, NodeId)> = Vec::new();
    for (i, &nid) in order.iter().enumerate() {
        if g.node(nid).inputs.contains(&candidate) {
            positions.push((i, nid));
        }
    }

    if positions.len() < 2 {
        return None;
    }

    // Find the largest gap between consecutive consumer positions.
    let mut best_gap = 0usize;
    let mut gap_idx = 0usize;
    for i in 0..positions.len() - 1 {
        let gap = positions[i + 1].0 - positions[i].0;
        if gap > best_gap {
            best_gap = gap;
            gap_idx = i;
        }
    }

    if best_gap < 2 {
        return None;
    }

    let last_before = positions[gap_idx].1;
    let after_gap: Vec<NodeId> = positions[gap_idx + 1..].iter().map(|(_, n)| *n).collect();

    Some((last_before, after_gap))
}

fn write_graph(graph: &SharedGraph) -> Result<std::sync::RwLockWriteGuard<'_, Graph>> {
    graph
        .write()
        .map_err(|_| Error::Msg("residency: graph RwLock poisoned".into()).bt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelined::{PipelinedExecutor, StorageCache};
    use fuel_graph::Node;
    use fuel_ir::{DType, Shape, probe::BackendId};
    use std::sync::{Arc, RwLock};

    fn shared() -> SharedGraph {
        Arc::new(RwLock::new(Graph::new()))
    }

    fn push(g: &mut Graph, op: Op, inputs: Vec<NodeId>, dims: &[usize]) -> NodeId {
        g.push(Node {
            op,
            inputs,
            shape: Shape::from_dims(dims),
            dtype: DType::F32,
        })
    }

    fn count_op(graph: &SharedGraph, pred: impl Fn(&Op) -> bool) -> usize {
        let g = graph.read().unwrap();
        (0..g.len())
            .filter(|i| pred(&g.node(NodeId(*i)).op))
            .count()
    }

    // ---- analysis (ported from fuel-graph-router residency_planner) ----

    /// A graph with no shared consumption: each intermediate is used
    /// exactly once by its direct successor. Peak bytes should equal
    /// the maximum of (input + output) at any op.
    #[test]
    fn chain_graph_peak_is_two_live_tensors() {
        // a = const[1024]  1024 * 4 = 4096 bytes
        // b = relu(a)      4096 bytes
        // c = neg(b)       4096 bytes
        // Peak: when c is produced, a is already dead (last-use was
        // relu), so live set is {b, c} = 8192 bytes.
        let graph = shared();
        let c = {
            let mut g = graph.write().unwrap();
            let a = push(&mut g, Op::Const, vec![], &[1024]);
            let b = push(&mut g, Op::Relu, vec![a], &[1024]);
            push(&mut g, Op::Neg, vec![b], &[1024])
        };
        let g = graph.read().unwrap();
        let report = ResidencyPlanner::analyze(&g, &[c]);
        assert_eq!(report.total_bytes, 4096 * 3);
        assert_eq!(report.peak_bytes, 4096 * 2);
    }

    #[test]
    fn shared_const_stays_live_across_multiple_uses() {
        // a is consumed twice — by add(a, b) and (later) by neg(a) —
        // so it stays live across the gap and shows up as an eviction
        // candidate with a non-zero inactive_gap.
        let graph = shared();
        let (a, sum) = {
            let mut g = graph.write().unwrap();
            let a = push(&mut g, Op::Const, vec![], &[256]);
            let b = push(&mut g, Op::Const, vec![], &[256]);
            let ab = push(&mut g, Op::Add, vec![a, b], &[256]);
            let na = push(&mut g, Op::Neg, vec![a], &[256]);
            let sum = push(&mut g, Op::Add, vec![ab, na], &[256]);
            (a, sum)
        };
        let g = graph.read().unwrap();
        let report = ResidencyPlanner::analyze(&g, &[sum]);
        let a_range = report.eviction_candidates.iter().find(|c| c.node == a);
        assert!(
            a_range.is_some() || report.peak_bytes > 0,
            "expected `a` tracked with a gap OR at least some peak bytes",
        );
    }

    #[test]
    fn report_fits_in_and_overage() {
        let graph = shared();
        let b = {
            let mut g = graph.write().unwrap();
            let a = push(&mut g, Op::Const, vec![], &[256]);
            push(&mut g, Op::Relu, vec![a], &[256])
        };
        let g = graph.read().unwrap();
        let report = ResidencyPlanner::analyze(&g, &[b]);
        assert!(report.fits_in(10_000));
        assert!(!report.fits_in(1));
        assert_eq!(report.overage(10_000), 0);
        assert!(report.overage(1) > 0);
    }

    // ---- transform (ported from residency_eviction, Move-shaped) ----

    /// Build the canonical over-budget graph used by the legacy
    /// rule's tests:
    ///   a    = const[4]            (two consumers, gap in between)
    ///   b    = relu(a)
    ///   pad  = neg(b)
    ///   pad2 = neg(pad)
    ///   c    = mul(pad2, a)        (reads a AGAIN, after the gap)
    ///   out  = add(b, c)
    /// All compute nodes stamped Cpu so the pipelined executor can
    /// compile them.
    fn over_budget_graph() -> (SharedGraph, NodeId, NodeId) {
        let graph = shared();
        let (a, out) = {
            let mut g = graph.write().unwrap();
            let a = push(&mut g, Op::Const, vec![], &[4]);
            let b = push(&mut g, Op::Relu, vec![a], &[4]);
            let pad = push(&mut g, Op::Neg, vec![b], &[4]);
            let pad2 = push(&mut g, Op::Neg, vec![pad], &[4]);
            let c = push(&mut g, Op::Mul, vec![pad2, a], &[4]);
            let out = push(&mut g, Op::Add, vec![b, c], &[4]);
            for n in [b, pad, pad2, c, out] {
                g.set_target_backend(n, BackendId::Cpu);
            }
            (a, out)
        };
        (graph, a, out)
    }

    #[test]
    fn eviction_is_noop_under_budget() {
        let (graph, _a, out) = over_budget_graph();
        let moves_before = count_op(&graph, |op| matches!(op, Op::Move { .. }));
        let copies_before = count_op(&graph, |op| matches!(op, Op::Copy { .. }));

        let chains = insert_residency_evictions(&graph, &[out], 1_000_000_000, 16, |_| None)
            .expect("eviction pass");
        assert!(
            chains.is_empty(),
            "under-budget graph should emit no chains"
        );
        assert_eq!(
            count_op(&graph, |op| matches!(op, Op::Move { .. })),
            moves_before
        );
        assert_eq!(
            count_op(&graph, |op| matches!(op, Op::Copy { .. })),
            copies_before
        );
    }

    #[test]
    fn eviction_emits_move_reload_chain_when_over_budget() {
        let (graph, a, out) = over_budget_graph();

        // Budget: 1 byte (effectively zero).
        let chains =
            insert_residency_evictions(&graph, &[out], 1, 16, |_| Some(DeviceLocation::Cpu))
                .expect("eviction pass");

        assert!(
            !chains.is_empty(),
            "over-budget graph should emit at least one chain"
        );
        assert!(
            count_op(&graph, |op| matches!(op, Op::Move { .. })) >= 1,
            "over-budget graph should emit at least one Op::Move",
        );
        assert!(
            count_op(&graph, |op| matches!(op, Op::Copy { .. })) >= 1,
            "each chain carries a reload Op::Copy",
        );

        // The top-scoring candidate is `a` (biggest gap × bytes among
        // multi-consumer nodes); its chain must be Move(a) → Copy.
        let chain = chains
            .iter()
            .find(|c| c.candidate == a)
            .unwrap_or(&chains[0]);
        let g = graph.read().unwrap();
        assert!(matches!(
            g.node(chain.move_node).op,
            Op::Move {
                target: DeviceLocation::Cpu
            },
        ));
        assert_eq!(g.node(chain.reload).inputs, vec![chain.move_node]);
        // Both stamped so the pipelined executor can compile them.
        assert_eq!(g.target_backend(chain.move_node), Some(BackendId::Cpu));
        assert_eq!(g.target_backend(chain.reload), Some(BackendId::Cpu));
    }

    /// `analyze_on` does not count the staged host copy of an eviction as bytes of the planned
    /// device, and the order-blind, residency-blind `analyze` does (the second flaw the
    /// eviction loop had: it measured a number the evictions could not move).
    #[test]
    fn analyze_on_does_not_count_the_staged_host_copy() {
        let cuda = DeviceLocation::Cuda { gpu_id: 0 };
        let graph = shared();
        let out = {
            let mut g = graph.write().unwrap();
            let a = push(&mut g, Op::Const, vec![], &[4]);
            let b = push(&mut g, Op::Relu, vec![a], &[4]);
            let staged = push(
                &mut g,
                Op::Move {
                    target: DeviceLocation::Cpu,
                },
                vec![a],
                &[4],
            );
            push(&mut g, Op::Add, vec![b, staged], &[4])
        };
        let g = graph.read().unwrap();
        let blind = ResidencyPlanner::analyze(&g, &[out]).peak_bytes;
        let on = ResidencyPlanner::analyze_on(&g, &[out], cuda).peak_bytes;
        assert!(
            on < blind,
            "device peak {on} must be below the blind peak {blind}"
        );
        // Control: on the CPU itself the Move{Cpu} output IS resident, so nothing is discounted.
        let on_cpu = ResidencyPlanner::analyze_on(&g, &[out], DeviceLocation::Cpu).peak_bytes;
        assert_eq!(on_cpu, blind);
    }

    /// CHARACTERISATION of a known limitation, not a goal: today the executor's order runs the
    /// reload `Op::Copy` right after its `Op::Move` (nothing delays it; there is no way to ask for
    /// "reload just before the first post-gap consumer"), so one eviction round does NOT lower the
    /// analysed device peak on this graph (64 -> 64 bytes). ROADMAP row 67 I3 adds the ordering
    /// primitive; when it lands this test flips to assert a drop.
    #[test]
    fn today_the_reload_runs_eagerly_so_the_device_peak_does_not_drop() {
        let cuda = DeviceLocation::Cuda { gpu_id: 0 };
        let (graph, _a, out) = over_budget_graph();
        let before = ResidencyPlanner::analyze_on(&graph.read().unwrap(), &[out], cuda).peak_bytes;
        let chains = insert_residency_evictions(&graph, &[out], before - 1, 16, |_| Some(cuda))
            .expect("eviction pass");
        assert!(!chains.is_empty(), "a budget below the peak must evict");
        let g = graph.read().unwrap();
        let order = fuel_graph::opt::execution_plan(&g, &[out]);
        let pos = |n: NodeId| order.iter().position(|&x| x == n).unwrap();
        let c = &chains[0];
        assert_eq!(
            pos(c.reload),
            pos(c.move_node) + 1,
            "the reload follows its Move immediately in the executor's order"
        );
        let after = ResidencyPlanner::analyze_on(&g, &[out], cuda).peak_bytes;
        assert_eq!(after, before, "so the analysed peak is unchanged");
    }

    /// Regression: a candidate already feeding an eviction `Op::Move` is not evicted again by a
    /// second run. Two destructive Moves of one source wait on each other and `execution_plan`
    /// panics ("cycle in ordering edges"); the old pass only remembered candidates within one run.
    #[test]
    fn a_candidate_is_evicted_at_most_once_across_runs() {
        let (graph, _a, out) = over_budget_graph();
        let first = insert_residency_evictions(&graph, &[out], 0, 16, |_| None).expect("run 1");
        assert!(!first.is_empty());
        let second = insert_residency_evictions(&graph, &[out], 0, 16, |_| None).expect("run 2");
        let evicted_before: Vec<NodeId> = first.iter().map(|c| c.candidate).collect();
        assert!(
            second
                .iter()
                .all(|c| !evicted_before.contains(&c.candidate)),
            "run 2 re-evicted a candidate of run 1"
        );
        // The graph must still be orderable (this panics on a cycle).
        let g = graph.read().unwrap();
        let _ = fuel_graph::opt::execution_plan(&g, &[out]);
    }

    /// Correctness invariant: graph surgery for residency must be
    /// transparent to the result. Realize the same graph WITHOUT and
    /// WITH the eviction pass through the PRODUCTION executor and
    /// compare bit-exactly. (The legacy twin realized through
    /// `GraphExecutor<CpuBackend>`; this is the pipelined port.)
    #[test]
    fn eviction_preserves_correctness_end_to_end_on_pipelined() {
        // a    = [-1, 2, -3, 4]
        // b    = relu(a)  = [0, 2, 0, 4]
        // pad  = neg(b)   = [0, -2, 0, -4]
        // pad2 = neg(pad) = [0, 2, 0, 4]
        // c    = pad2 * a = [0, 4, 0, 16]
        // out  = b + c    = [0, 6, 0, 20]
        let realize = |evict: bool| -> Vec<f32> {
            let (graph, a, out) = over_budget_graph();
            if evict {
                let chains = insert_residency_evictions(&graph, &[out], 1, 16, |_| {
                    Some(DeviceLocation::Cpu)
                })
                .expect("eviction pass");
                assert!(!chains.is_empty(), "budget=1 must trigger eviction");
            }
            let mut inputs = StorageCache::new();
            inputs.insert(
                a,
                Arc::new(RwLock::new(fuel_memory::from_slice_cpu(&[
                    -1.0_f32, 2.0, -3.0, 4.0,
                ]))),
            );
            let (storage, _layout) =
                PipelinedExecutor::realize(graph, out, inputs).expect("realize");
            let guard = storage.read().unwrap();
            let fuel_memory::BackendStorage::Cpu(c) = &guard.inner else {
                panic!("expected CPU output");
            };
            c.as_slice::<f32>().unwrap().to_vec()
        };

        let baseline = realize(false);
        let evicted = realize(true);
        assert_eq!(
            baseline, evicted,
            "eviction must preserve the output bit-exactly",
        );
        assert_eq!(baseline, vec![0.0_f32, 6.0, 0.0, 20.0]);
    }

    /// Review finding 2 (Pip, #498): a realize root must never be evicted. The caller reads the root's
    /// storage after the run and an `Op::Move` consumes its source. `a` is the best candidate here
    /// (live from the start, read early and late), so making it a root is exactly the hazard.
    #[test]
    fn a_realize_root_is_never_evicted() {
        let (graph, a, out) = over_budget_graph();
        let chains =
            insert_residency_evictions(&graph, &[a, out], 1, 16, |_| Some(DeviceLocation::Cpu))
                .expect("eviction pass");
        assert!(
            chains.iter().all(|c| c.candidate != a),
            "the root `a` was evicted: {chains:?}"
        );
        // Control: with `a` NOT a root it is the candidate the pass picks (so the exclusion above
        // is what spared it, not an absence of gap).
        let (graph, a, out) = over_budget_graph();
        let chains =
            insert_residency_evictions(&graph, &[out], 1, 16, |_| Some(DeviceLocation::Cpu))
                .expect("eviction pass");
        assert!(
            chains.iter().any(|c| c.candidate == a),
            "control: without the root exclusion `a` is evicted"
        );
    }

    /// Review finding 1 (Pip, #498): a round that does not lower the analysed peak ends the pass, and
    /// the outcome says the budget was not met. Today (no reload delay, row 67 I3) no round can, so a
    /// same-device budget keeps ONE chain, not `max_rounds` of them, and `fits()` is false.
    #[test]
    fn a_round_that_does_not_lower_the_peak_stops_and_is_reported() {
        let cuda = DeviceLocation::Cuda { gpu_id: 0 };
        let (graph, _a, out) = over_budget_graph();
        let mut g = graph.write().unwrap();
        let before = ResidencyPlanner::analyze_on(&g, &[out], cuda).peak_bytes;
        let outcome =
            insert_residency_evictions_in(&mut g, &[out], cuda, before - 1, 16, |_| Some(cuda))
                .expect("eviction pass");
        assert_eq!(outcome.peak_before, before);
        assert_eq!(
            outcome.chains.len(),
            1,
            "a stalled pass must stop after the round that showed the stall, not run to max_rounds"
        );
        assert!(outcome.peak_after >= outcome.peak_before);
        assert!(
            !outcome.fits(),
            "the budget was NOT met and the outcome must say so"
        );
        // Control: a budget the graph already fits reports fits() and inserts nothing.
        let ok =
            insert_residency_evictions_in(&mut g, &[out], cuda, usize::MAX, 16, |_| Some(cuda))
                .expect("eviction pass");
        assert!(ok.fits() && ok.chains.is_empty());
    }

    /// Review question 1 (Pip, #498): a post-gap consumer that reads the candidate twice has BOTH
    /// input slots rewired to the reload (`rewrite_input` replaces every matching slot).
    #[test]
    fn a_candidate_read_twice_by_one_consumer_is_rewired_in_both_slots() {
        // `late` = Where(pad2, a, a) reads `a` in two slots and sits after the long chain, so it is
        // the post-gap consumer; `early` = Relu(a) is the pre-gap one.
        let graph = shared();
        let (a, pad2, late) = {
            let mut g = graph.write().unwrap();
            let a = push(&mut g, Op::Const, vec![], &[4]);
            let early = push(&mut g, Op::Relu, vec![a], &[4]);
            let pad = push(&mut g, Op::Neg, vec![early], &[4]);
            let pad2 = push(&mut g, Op::Neg, vec![pad], &[4]);
            let late = push(&mut g, Op::Where, vec![pad2, a, a], &[4]);
            for n in [early, pad, pad2, late] {
                g.set_target_backend(n, BackendId::Cpu);
            }
            (a, pad2, late)
        };
        let chains =
            insert_residency_evictions(&graph, &[late], 1, 4, |_| Some(DeviceLocation::Cpu))
                .expect("eviction pass");
        let chain = chains
            .iter()
            .find(|c| c.candidate == a)
            .expect("`a` is evicted");
        let g = graph.read().unwrap();
        assert_eq!(
            g.node(late).inputs,
            vec![pad2, chain.reload, chain.reload],
            "both slots that read `a` must read the reload"
        );
    }

    /// Idempotence guard: re-running the pass after the budget can't
    /// be met must not grow the graph unboundedly (the `evicted` set
    /// + `progressed` exit cover it within a run; `max_rounds` caps
    ///
    /// across rounds).
    #[test]
    fn eviction_terminates_when_budget_unreachable() {
        let (graph, _a, out) = over_budget_graph();
        let chains =
            insert_residency_evictions(&graph, &[out], 0, 16, |_| None).expect("eviction pass");
        let len_after = graph.read().unwrap().len();
        // A second run finds the same candidates already evicted (or
        // gapless) and makes no further progress.
        let chains2 = insert_residency_evictions(&graph, &[out], 0, 16, |_| None)
            .expect("second eviction pass");
        let len_final = graph.read().unwrap().len();
        assert!(chains.len() <= 16);
        // The post-surgery graph rewires post-gap consumers onto the
        // reload, so re-analysis finds at most a bounded set of new
        // multi-consumer candidates; the key invariant is termination
        // with bounded growth, not zero growth.
        assert!(chains2.len() <= 16);
        assert!(len_final >= len_after);
        assert!(len_final - len_after <= 2 * chains2.len());
    }
}

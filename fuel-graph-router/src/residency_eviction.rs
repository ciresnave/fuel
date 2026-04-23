//! Residency eviction rule — the first [`GraphMutatingSchedulerRule`].
//!
//! Uses the [`ResidencyPlanner`] analysis to identify tensors whose
//! continued residency would push the graph's peak memory over a budget,
//! and inserts `Op::Copy{Cpu}` + `Op::Release` + `Op::Copy{device}`
//! chains to spill-and-restore the data around its longest
//! inactive-gap window.
//!
//! The `derive_ordering` pass then automatically pins each emitted
//! `Op::Release` to run after all non-destructive readers of the
//! evicted tensor, and the executor frees the device storage when the
//! Release runs.
//!
//! ## Today's scope
//!
//! For each candidate (picked in order by `bytes × inactive_gap`
//! desc), the rule:
//!
//! 1. Finds all existing consumers of the candidate in topo order.
//! 2. Identifies the largest gap between consecutive reads.
//! 3. If the gap is ≥ 2 op positions, emits an evict-chain:
//!    - `Op::Copy{Cpu}` reading the candidate (stages to CPU)
//!    - `Op::Release` destroying the candidate on its device
//!    - `Op::Copy{device}` reading the CPU copy (reload)
//!    Post-gap consumers get their `candidate` input rewritten to
//!    read from the reload.
//! 4. Re-analyzes and continues picking candidates until the peak fits
//!    under the budget, or no more candidates have a meaningful gap.
//!
//! ## Known limitations (follow-up work)
//!
//! - Only handles one eviction per candidate (the largest gap). Multiple
//!   gaps would need multiple evict chains.
//! - Simple greedy selection; no cost-awareness for the CPU↔device
//!   transfer overhead vs the VRAM savings.
//! - Doesn't track cumulative re-upload costs when a weight is evicted
//!   and later faulted back into VRAM multiple times.
//! - No integration with the `VulkanBackend::evict` mmap-tiering path
//!   — that's a transparent runtime mechanism, distinct from
//!   graph-level planning. The unified durable-tensor-store design
//!   (north-star) would eventually collapse both.

use fuel_core_types::DeviceLocation;
use fuel_graph::{opt, topo_order_multi, NodeId, SharedGraph};

use crate::{
    scheduler::{GraphMutatingSchedulerRule, Placement},
    residency_planner::ResidencyPlanner,
    Router,
};

/// Graph-mutating scheduler rule that emits evict/reload chains for
/// tensors whose continued residency would exceed a byte budget.
///
/// Construct with [`ResidencyEvictionRule::new(budget_bytes)`] and add
/// to a [`RuleScheduler`](crate::RuleScheduler) via
/// [`with_mutating_rule`](crate::RuleScheduler::with_mutating_rule).
pub struct ResidencyEvictionRule {
    budget_bytes: usize,
    /// Cap on number of eviction rounds to prevent pathological loops.
    max_rounds: usize,
}

impl ResidencyEvictionRule {
    pub fn new(budget_bytes: usize) -> Self {
        Self { budget_bytes, max_rounds: 16 }
    }

    /// Change the maximum number of eviction rounds (default 16).
    pub fn with_max_rounds(mut self, n: usize) -> Self {
        self.max_rounds = n;
        self
    }
}

impl GraphMutatingSchedulerRule for ResidencyEvictionRule {
    fn apply(
        &self,
        graph: &SharedGraph,
        roots: &[NodeId],
        _router: &Router,
        placement: &mut Placement,
    ) {
        // Track candidates we've already evicted this run. Without this
        // guard, a just-evicted candidate's new consumers (its own
        // cpu_copy and release) show up as post-gap consumers in the
        // next round, triggering infinite re-eviction.
        let mut evicted: std::collections::HashSet<NodeId> = std::collections::HashSet::new();

        for _round in 0..self.max_rounds {
            let report = ResidencyPlanner::analyze(graph, roots);
            if report.fits_in(self.budget_bytes) {
                return;
            }
            // Pick the top candidate that still has a realizable gap
            // AND hasn't been evicted already in this run.
            let mut progressed = false;
            for cand in &report.eviction_candidates {
                if evicted.contains(&cand.node) { continue; }

                let Some((_pre_gap, post_gap_consumers)) =
                    find_gap_consumers(graph, cand.node) else { continue };

                if post_gap_consumers.is_empty() { continue; }

                let src_device = placement
                    .get(&cand.node)
                    .copied()
                    .unwrap_or(DeviceLocation::Cpu);

                let (cpu_copy_id, _release_id, reload_id) = opt::insert_evict_reload(
                    graph, cand.node, src_device, &post_gap_consumers,
                );

                placement.insert(cpu_copy_id, DeviceLocation::Cpu);
                placement.insert(reload_id, src_device);
                evicted.insert(cand.node);

                progressed = true;
                break;
            }
            if !progressed {
                return;
            }
        }
    }
}

/// Find the pre-gap and post-gap consumers of `candidate` — i.e., the
/// consumer that reads it just before its longest inactive stretch,
/// and the consumer(s) that read it after.
///
/// Returns `None` if the candidate has fewer than two consumers, or if
/// the largest gap is < 2 op positions (no meaningful window to evict
/// into).
fn find_gap_consumers(
    graph: &SharedGraph,
    candidate: NodeId,
) -> Option<(NodeId, Vec<NodeId>)> {
    let g = graph.borrow();
    let order = topo_order_multi(&g, &collect_roots(&g));

    // Positions of candidate's consumers in topo order.
    let mut positions: Vec<(usize, NodeId)> = Vec::new();
    for (i, &nid) in order.iter().enumerate() {
        if g.node(nid).inputs.iter().any(|&inp| inp == candidate) {
            positions.push((i, nid));
        }
    }

    if positions.len() < 2 { return None; }

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

    if best_gap < 2 { return None; }

    let last_before = positions[gap_idx].1;
    let after_gap: Vec<NodeId> = positions[gap_idx + 1..]
        .iter()
        .map(|(_, n)| *n)
        .collect();

    Some((last_before, after_gap))
}

/// Collect every NodeId in the graph as a synthetic "root set" for
/// topo walking. We use this instead of the real roots because the
/// rule needs to see ALL consumers of a candidate, not just those on
/// a specific root's path.
fn collect_roots(g: &fuel_graph::Graph) -> Vec<NodeId> {
    (0..g.len()).map(NodeId).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core_types::Shape;
    use fuel_graph::{Op, Tensor};

    /// Helper: count how many Op::Release nodes are in the graph.
    fn count_releases(graph: &SharedGraph) -> usize {
        let g = graph.borrow();
        (0..g.len())
            .filter(|i| matches!(g.node(NodeId(*i)).op, Op::Release))
            .count()
    }

    fn count_copies(graph: &SharedGraph) -> usize {
        let g = graph.borrow();
        (0..g.len())
            .filter(|i| matches!(g.node(NodeId(*i)).op, Op::Copy { .. }))
            .count()
    }

    #[test]
    fn rule_is_noop_under_budget() {
        // Tiny graph, huge budget — rule should not mutate.
        let a = Tensor::from_f32(vec![1.0_f32; 4], Shape::from_dims(&[4]));
        let b = a.relu();
        let c = a.neg();
        let out = b.add(&c);
        let graph = a.graph().clone();

        let rule = ResidencyEvictionRule::new(1_000_000_000);
        let router = Router::new().add_cpu();
        let mut placement = Placement::new();

        let releases_before = count_releases(&graph);
        let copies_before = count_copies(&graph);
        rule.apply(&graph, &[out.id()], &router, &mut placement);
        assert_eq!(count_releases(&graph), releases_before,
            "under-budget graph should emit no Release");
        assert_eq!(count_copies(&graph), copies_before,
            "under-budget graph should emit no Copy");
    }

    #[test]
    fn rule_emits_evict_chain_when_over_budget() {
        // `a` has two consumers, and the second is transitively
        // dependent on intermediate ops that DON'T read a. That
        // guarantees the natural topo order puts those ops between
        // a's first reader and a's second reader (creating a gap the
        // rule can exploit).
        //
        // Graph (canonical topo order):
        //   a = const[1024]            (pos 0)
        //   b = relu(a)                (pos 1 — reads a)
        //   pad = neg(b)               (pos 2)
        //   pad2 = neg(pad)            (pos 3)
        //   c = mul(pad2, a)           (pos 4 — reads a AGAIN, after gap)
        //   out = add(b, c)            (pos 5)
        let a = Tensor::from_f32(vec![1.0_f32; 1024], Shape::from_dims(&[1024]));
        let b = a.relu();
        let pad = b.neg();
        let pad2 = pad.neg();
        let c = pad2.mul(&a); // c depends on pad2 AND a — topo pins c after pad2
        let out = b.add(&c);
        let graph = a.graph().clone();

        // Budget: 1 byte (effectively zero).
        let rule = ResidencyEvictionRule::new(1);
        let router = Router::new().add_cpu();
        let mut placement = Placement::new();
        placement.insert(a.id(), DeviceLocation::Cpu);

        rule.apply(&graph, &[out.id()], &router, &mut placement);

        assert!(count_releases(&graph) >= 1,
            "over-budget graph should emit at least one Release");
        // Two new Copies per evict: cpu_copy + reload.
        assert!(count_copies(&graph) >= 2,
            "over-budget graph should emit at least two Copies");
    }

    #[test]
    fn rule_preserves_correctness_end_to_end() {
        // Build the same over-budget graph. Realize WITHOUT the rule
        // and WITH the rule. Both should produce identical output.
        // This is the correctness invariant: graph surgery for
        // residency must be transparent to the result.
        use fuel_graph_executor::GraphExecutor;

        // a    = [-1, 2, -3, 4]
        // b    = relu(a)  = [0, 2, 0, 4]
        // pad  = neg(b)   = [0, -2, 0, -4]
        // pad2 = neg(pad) = [0, 2, 0, 4]
        // c    = pad2 * a = [0, 4, 0, 16]
        // out  = b + c    = [0, 6, 0, 20]
        let make_graph = || {
            let a = Tensor::from_f32(
                vec![-1.0_f32, 2.0, -3.0, 4.0], Shape::from_dims(&[4]));
            let b = a.relu();
            let pad = b.neg();
            let pad2 = pad.neg();
            let c = pad2.mul(&a);
            (a, b.add(&c))
        };

        // Baseline: no eviction.
        let (_a1, out1) = make_graph();
        let mut exec1: GraphExecutor<fuel_graph_cpu::CpuBackend> =
            GraphExecutor::new(fuel_graph_cpu::CpuBackend);
        let r1 = exec1.realize_f32(&out1);

        // With eviction.
        let (a2, out2) = make_graph();
        let graph2 = a2.graph().clone();
        let rule = ResidencyEvictionRule::new(1);
        let router = Router::new().add_cpu();
        let mut placement = Placement::new();
        placement.insert(a2.id(), DeviceLocation::Cpu);
        rule.apply(&graph2, &[out2.id()], &router, &mut placement);

        let mut exec2: GraphExecutor<fuel_graph_cpu::CpuBackend> =
            GraphExecutor::new(fuel_graph_cpu::CpuBackend);
        let r2 = exec2.realize_f32(&out2);

        assert_eq!(r1.as_slice(), r2.as_slice(),
            "eviction must preserve the output bit-exactly");
        assert_eq!(r1.as_slice(), &[0.0_f32, 6.0, 0.0, 20.0]);
    }
}

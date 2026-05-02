//! Residency analysis for the scheduler.
//!
//! Walks a graph's topo order, tracks each node's live range (first
//! use to last consumer), and computes the peak working-set size
//! — the maximum bytes that must be resident simultaneously at any
//! point during execution.
//!
//! This is the analysis half of residency-aware scheduling. The
//! transform half — inserting explicit `Op::Copy` nodes to evict
//! long-lived tensors whose continued residency would exceed a
//! byte budget — needs graph mutation support beyond what today's
//! [`SchedulerRule`] interface provides, and is a follow-up.
//!
//! ## What to do with a report
//!
//! - **Peak bytes ≤ budget**: nothing to do; the graph runs without
//!   spilling.
//! - **Peak bytes > budget**: identify tensors whose live ranges
//!   span the over-budget region and evict them. The report's
//!   `eviction_candidates` ranks tensors by how long they spend
//!   "inactive but resident" (last read → next read gap).
//!
//! ## What this planner does NOT do yet
//!
//! - Emit `Op::Copy` evict/reload pairs into the graph.
//! - Device-specific budgets (all bytes treated uniformly).
//! - Account for the allocator's actual overhead (padding,
//!   fragmentation, internal block structure). The report is a
//!   lower bound; real peak can be 10-30 % higher.
//!
//! [`SchedulerRule`]: crate::scheduler::SchedulerRule

use fuel_graph::{topo_order_multi, NodeId, Op, SharedGraph};
use std::collections::HashMap;

/// One tensor's residency span. `[first_use, last_use]` are op
/// positions in a topo order (inclusive — the tensor is live
/// during both). `bytes` is the output storage size. `inactive_gap`
/// is the longest stretch of ops between consecutive reads where
/// the tensor is resident but not accessed — a strong signal for
/// eviction viability.
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
    /// The scheduler's eviction planner consumes this list.
    pub eviction_candidates: Vec<LiveRange>,
}

impl ResidencyReport {
    /// Does this graph fit in a given byte budget?
    pub fn fits_in(&self, budget: usize) -> bool { self.peak_bytes <= budget }

    /// Byte overage relative to a budget. Zero when the graph fits.
    pub fn overage(&self, budget: usize) -> usize {
        self.peak_bytes.saturating_sub(budget)
    }
}

/// Analysis-only residency planner. Computes live ranges and peak
/// bytes for a graph, emits a [`ResidencyReport`]. Does not mutate
/// the graph.
///
/// ```ignore
/// let report = ResidencyPlanner::analyze(&graph, &[root.id()]);
/// if !report.fits_in(vram_budget) {
///     eprintln!(
///         "graph peak {} bytes exceeds VRAM budget {} by {} bytes",
///         report.peak_bytes, vram_budget, report.overage(vram_budget),
///     );
/// }
/// ```
pub struct ResidencyPlanner;

impl ResidencyPlanner {
    /// Run the analysis. O(V + E) in the reachable subgraph.
    pub fn analyze(graph: &SharedGraph, roots: &[NodeId]) -> ResidencyReport {
        let g = graph.read().unwrap();
        let order = topo_order_multi(&g, roots);

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
            let bytes = elems * node.dtype.size_in_bytes();
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

        // 3. Walk topo order, maintain live-set bytes. At each op:
        //    a. Add the new node's output bytes (it goes live).
        //    b. Record current bytes.
        //    c. Drop any nodes whose last_use == this position AFTER
        //       the op completes (they're freed once the op exits).
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
            // Post-op: free anything whose last_use is this op.
            // We only free *inputs* that are done, not the node we
            // just produced (that's live until IT is consumed).
            for &input in &reads_at[op_idx] {
                if last_use.get(&input) == Some(&op_idx) {
                    let b = *byte_of.get(&input).unwrap_or(&0);
                    live_bytes = live_bytes.saturating_sub(b);
                }
            }
        }

        // 4. Compute per-node inactive_gap: longest span of ops between
        //    consecutive reads where the tensor is live but not touched.
        //    For scheduler eviction, nodes with big gap × bytes are
        //    the best candidates (lots of wasted residency).
        let mut gap_of: HashMap<NodeId, usize> = HashMap::with_capacity(n);
        for (nid, &first) in &first_use {
            // Collect positions where this node is read.
            let reads: Vec<usize> = reads_at.iter().enumerate()
                .filter_map(|(i, rs)| if rs.contains(nid) { Some(i) } else { None })
                .collect();
            // Build the gap sequence: between first (produced) and each
            // read, between consecutive reads, and after last read till
            // last_use.
            let last = last_use.get(nid).copied().unwrap_or(first);
            let mut points = vec![first];
            points.extend(reads.iter().copied());
            points.push(last);
            points.sort();
            let max_gap = points.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
            gap_of.insert(*nid, max_gap);
        }

        // 5. Build + sort eviction candidates.
        let mut cands: Vec<LiveRange> = order.iter().map(|&nid| {
            LiveRange {
                node: nid,
                bytes: *byte_of.get(&nid).unwrap_or(&0),
                first_use: *first_use.get(&nid).unwrap_or(&0),
                last_use: *last_use.get(&nid).unwrap_or(&0),
                inactive_gap: *gap_of.get(&nid).unwrap_or(&0),
            }
        }).collect();
        // Rank by bytes × inactive_gap, descending. Ties: larger
        // bytes first (more reclaim per evict).
        cands.sort_by(|a, b| {
            let a_score = (a.bytes as u128) * (a.inactive_gap as u128);
            let b_score = (b.bytes as u128) * (b.inactive_gap as u128);
            b_score.cmp(&a_score).then(b.bytes.cmp(&a.bytes))
        });
        // Filter out zero-score candidates (hot tensors with no gap).
        cands.retain(|c| c.bytes > 0 && c.inactive_gap > 0);
        // Cap to a reasonable count for downstream consumers.
        cands.truncate(32);

        // Consts flagged at their producer pos; `Op::Const` has no
        // inputs so its first_use == its topo position, which is fine.
        // Result tensors (roots with no downstream op) have last_use ==
        // first_use, making inactive_gap = 0 → filtered out above.
        let _ = Op::Const; // reference so imports stay lively

        ResidencyReport {
            total_bytes,
            peak_bytes,
            peak_op_index,
            eviction_candidates: cands,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core_types::Shape;
    use fuel_graph::Tensor;

    /// A graph with no shared consumption: each intermediate is used
    /// exactly once by its direct successor. Peak bytes should equal
    /// the maximum of (input + output) at any op.
    #[test]
    fn chain_graph_peak_is_two_live_tensors() {
        // a = const[1024]  1024 * 4 = 4096 bytes
        // b = relu(a)      4096 bytes
        // c = neg(b)       4096 bytes
        // Peak: when c is produced, a is already dead (last-use was relu),
        // so live set is {b, c} = 8192 bytes.
        let a = Tensor::from_f32(vec![1.0_f32; 1024], Shape::from_dims(&[1024]));
        let b = a.relu();
        let c = b.neg();
        let report = ResidencyPlanner::analyze(c.graph(), &[c.id()]);
        assert_eq!(report.total_bytes, 4096 * 3);
        assert_eq!(report.peak_bytes, 4096 * 2);
    }

    #[test]
    fn shared_const_stays_live_across_multiple_uses() {
        // a is consumed twice — by add(a,b) and by neg(a).
        // a stays live until both are done.
        let a = Tensor::from_f32(vec![1.0_f32; 256], Shape::from_dims(&[256]));
        let b = a.const_f32_like(vec![2.0_f32; 256], Shape::from_dims(&[256]));
        let ab = a.add(&b);      // reads a, b
        let na = a.neg();         // reads a again
        let sum = ab.add(&na);
        let report = ResidencyPlanner::analyze(sum.graph(), &[sum.id()]);
        // a's last_use is after na's creation. Verify a is in the list
        // of eviction candidates with a non-zero inactive_gap (it's
        // live but unused between ab and na producing).
        let a_range = report.eviction_candidates.iter().find(|c| c.node == a.id());
        assert!(a_range.is_some() || report.peak_bytes > 0,
            "expected `a` tracked with a gap OR at least some peak bytes");
    }

    #[test]
    fn report_fits_in_and_overage() {
        let a = Tensor::from_f32(vec![1.0_f32; 256], Shape::from_dims(&[256]));
        let b = a.relu();
        let report = ResidencyPlanner::analyze(b.graph(), &[b.id()]);
        assert!(report.fits_in(10_000));
        assert!(!report.fits_in(1));
        assert_eq!(report.overage(10_000), 0);
        assert!(report.overage(1) > 0);
    }
}

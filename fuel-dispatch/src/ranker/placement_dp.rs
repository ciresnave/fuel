//! Carry-forward placement DP — planner Stage 3
//! (`docs/session-prompts/load-time-incremental-planner.md`,
//! architecture `04-optimization` §Load-time incremental planning,
//! design pillars 3+4).
//!
//! Stage 2's greedy per-node ranking commits each node's winner
//! given its producers' already-committed placements. That is
//! conservative-correct for under-moving (it never makes an
//! unjustified move) but has two structural biases:
//!
//! - **Under-moves**: the first op of a beneficial migration must
//!   justify the device crossing alone — a segment of ops whose
//!   *combined* win exceeds the crossing never migrates.
//! - **Over-moves / stranding**: a locally-fast op (fused or not)
//!   that wins on inbound pricing can strand residency that a later
//!   segment needs back on the original device; the exit crossing
//!   is unpriced at the op's own decision point.
//!
//! The DP fixes both by carrying accumulated cost forward instead of
//! committing per node:
//!
//! - **State**: per frontier node, `best[d]` = lowest accumulated
//!   wall-clock cost arriving here with output resident on device
//!   `d`, for each device `d` that has at least one admissible
//!   candidate at the node (devices with no candidate are absent
//!   from the row).
//! - **Extension**: `best_N[d] = min over d' of (best_{N-1}[d'] +
//!   transfer(d'→d, boundary bytes) + cost(N on d))`, where `cost`
//!   is the existing composite (Layer-1 static, Layer-2
//!   Judge-refined) plus per-candidate inbound terms for inputs
//!   whose residency is KNOWN-fixed at plan time (graph inputs,
//!   consts, already-committed nodes).
//! - **Exit**: terminal (still-open) states price
//!   `transfer(d → realize_target)` so the return crossing greedy
//!   ignored participates in the final argmin.
//! - **Commit**: backtracking from the best terminal state stamps
//!   one device per row; `compile_plan` prunes each row's
//!   `AlternativeSet` to that device, preserving the Stage-2
//!   "surviving set lives on ONE device" invariant the residency
//!   stitch depends on.
//!
//! ## Fused jumps
//!
//! Fused ops are single graph nodes (`Op::Fused(FusedOpId, …)`), so
//! they enter the recurrence as single states automatically — a
//! locally-fast fused kernel that strands residency loses on
//! accumulated + exit cost exactly like a primitive would. Pattern
//! probing for *new* fusions during planning is Stage-4 driver
//! territory, not this module's.
//!
//! ## Joins, fan-out, and the documented debt
//!
//! The recurrence is exact for the dominant chain of the topo
//! order. DAG shapes are handled heuristically (architecture doc:
//! "branch handling may start heuristic — merge to the cheaper
//! producer device — and tighten later"):
//!
//! - **Joins** (a node with several open-row producers): every open
//!   producer contributes `min over d' of (best_p[d'] +
//!   transfer(d'→d))` independently per consumer device `d`, each
//!   with its own backpointer. This is the "cheaper producer
//!   device" merge, applied per arriving device. The producers'
//!   choices are treated as independent given the consumer's
//!   device, which is exact for tree-shaped prefixes and heuristic
//!   when branches share ancestors.
//! - **Fan-out** (an open row consumed by several nodes): the FIRST
//!   consumer in topo order chains the row (closes it). Later
//!   consumers see the row as residency-unknown and price no term
//!   for that edge — conservative (an unpriceable edge neither
//!   justifies nor penalizes a move), and the realize-level
//!   correctness is unaffected because the bridge's residency
//!   stitch inserts `Op::Copy` from the *final* committed
//!   placements. Only optimality is at stake; tighten in Stage 4+.
//! - **Diamonds**: the two branches chain independently (the second
//!   branch starts a fresh chain at the shared ancestor's fan-out),
//!   merge at the join per the rule above, and commit consistently
//!   via backpointers — see `compile_plan`'s diamond test.
//!
//! ## Complexity
//!
//! Per node: O(producers × devices²) extension work with devices
//! bounded by the topology (≤ 4 inline). Whole plan: O(nodes ×
//! devices²). The 1000-node-chain perf test in `plan.rs` keeps this
//! honest.
//!
//! No panics on production paths: defensive branches degrade to
//! "commit at the row's own cheapest state" rather than unwrap.

use std::collections::HashMap;

use fuel_ir::DeviceLocation;
use fuel_graph::NodeId;
use smallvec::SmallVec;

use super::cost::TransferEstimator;

/// Inline capacity for per-row device state. CPU + CUDA + Vulkan +
/// Metal is the realistic ceiling today; spill is harmless.
type DeviceVec = SmallVec<[DeviceLocation; 4]>;
type CostVec = SmallVec<[u64; 4]>;

/// One chain edge group feeding a new row: an open producer row plus
/// the boundary bytes of every graph edge from it into the consumer
/// (duplicated inputs — `Add(x, x)` — carry one entry per edge,
/// matching the Stage-2 convention of pricing per input occurrence).
#[derive(Clone, Debug)]
pub struct ChainInput {
    /// The producer's row-bearing node (view aliases pre-resolved by
    /// the caller).
    pub producer: NodeId,
    /// Boundary bytes per edge from `producer` into the consumer.
    pub edge_bytes: SmallVec<[u64; 2]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowStatus {
    /// Frontier state — extendable by a consumer, exit-priced at
    /// `finish` if nothing consumes it.
    Open,
    /// Chained into a downstream row; commits via that row's
    /// backpointers.
    Chained,
    /// Device decided.
    Committed,
}

#[derive(Clone, Debug)]
struct DpRow {
    node: NodeId,
    /// Candidate devices, in first-seen candidate order (decision
    /// device first — ties in every argmin break toward locality).
    devices: DeviceVec,
    /// `best[i]` = lowest accumulated cost arriving at this node
    /// with output on `devices[i]`. Saturating arithmetic.
    best: CostVec,
    /// `back[i]` = per-producer device choices made when arriving on
    /// `devices[i]`.
    back: Vec<Vec<(NodeId, DeviceLocation)>>,
    status: RowStatus,
    committed: Option<DeviceLocation>,
}

/// The DP table `compile_plan` threads along its topo walk. See the
/// module docs for the recurrence and the heuristics.
#[derive(Debug, Default)]
pub struct PlacementDp {
    rows: Vec<DpRow>,
    by_node: HashMap<NodeId, usize>,
    /// View-shaped pass-through aliases (view node → row node),
    /// flattened at insertion so resolution is O(1).
    aliases: HashMap<NodeId, NodeId>,
}

impl PlacementDp {
    pub fn new() -> Self {
        Self::default()
    }

    /// True when no rows were ever opened — single-device plans pay
    /// only this check.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Resolve `id` through the view-alias map to its row-bearing
    /// node (or itself when no alias exists).
    pub fn resolve(&self, id: NodeId) -> NodeId {
        self.aliases.get(&id).copied().unwrap_or(id)
    }

    /// Record a view-shaped pass-through: `view`'s bytes are
    /// `src`'s bytes, so chain edges through `view` connect to
    /// `src`'s row. `src` is resolved first (flattening), so chains
    /// of views stay O(1) to resolve.
    pub fn add_alias(&mut self, view: NodeId, src: NodeId) {
        let root = self.resolve(src);
        self.aliases.insert(view, root);
    }

    /// Whether `id` (already alias-resolved) has an OPEN row.
    pub fn is_open(&self, id: NodeId) -> bool {
        self.by_node
            .get(&id)
            .is_some_and(|&i| self.rows[i].status == RowStatus::Open)
    }

    /// The committed device for `id`'s row, if the row exists and
    /// has been committed.
    pub fn committed_device(&self, id: NodeId) -> Option<DeviceLocation> {
        self.by_node.get(&id).and_then(|&i| self.rows[i].committed)
    }

    /// Open a new row for `node`.
    ///
    /// - `device_costs` — per candidate device, the node's own cost
    ///   on that device (min composite over same-device candidates).
    ///   Order is preserved; put the decision device first so ties
    ///   break toward locality.
    /// - `fixed_inputs` — `(residency, bytes)` per input edge whose
    ///   residency is known-fixed at plan time.
    /// - `chain_inputs` — open producer rows feeding this node; each
    ///   is extended via the DP recurrence and marked `Chained`.
    pub fn push_row(
        &mut self,
        node: NodeId,
        device_costs: &[(DeviceLocation, u64)],
        fixed_inputs: &[(DeviceLocation, u64)],
        chain_inputs: &[ChainInput],
        est: &dyn TransferEstimator,
    ) {
        let mut devices = DeviceVec::new();
        let mut best = CostVec::new();
        let mut back: Vec<Vec<(NodeId, DeviceLocation)>> =
            Vec::with_capacity(device_costs.len());

        for &(d, kernel_ns) in device_costs {
            let mut acc = kernel_ns;
            for &(src, bytes) in fixed_inputs {
                acc = acc.saturating_add(est.estimate_transfer_ns(src, d, bytes));
            }
            let mut bp = Vec::with_capacity(chain_inputs.len());
            for ci in chain_inputs {
                // Producer must have a row (caller checked is_open);
                // defensive skip otherwise — an unpriceable edge.
                let Some(&p_idx) = self.by_node.get(&ci.producer) else {
                    continue;
                };
                let p = &self.rows[p_idx];
                let mut choice: Option<(DeviceLocation, u64)> = None;
                for (j, &pd) in p.devices.iter().enumerate() {
                    let mut t = p.best[j];
                    for &b in &ci.edge_bytes {
                        t = t.saturating_add(est.estimate_transfer_ns(pd, d, b));
                    }
                    if choice.is_none_or(|(_, c)| t < c) {
                        choice = Some((pd, t));
                    }
                }
                if let Some((pd, c)) = choice {
                    acc = acc.saturating_add(c);
                    bp.push((ci.producer, pd));
                }
            }
            devices.push(d);
            best.push(acc);
            back.push(bp);
        }

        // The consumed producers are no longer frontier states.
        for ci in chain_inputs {
            if let Some(&p_idx) = self.by_node.get(&ci.producer) {
                if self.rows[p_idx].status == RowStatus::Open {
                    self.rows[p_idx].status = RowStatus::Chained;
                }
            }
        }

        let idx = self.rows.len();
        self.rows.push(DpRow {
            node,
            devices,
            best,
            back,
            status: RowStatus::Open,
            committed: None,
        });
        self.by_node.insert(node, idx);
    }

    /// Close an OPEN row toward a consumer whose device is already
    /// decided (greedy-finalized nodes, `Op::Copy`/`Op::Move`
    /// targets): commit the row at
    /// `argmin over d' of (best[d'] + Σ transfer(d'→consumer, bytes))`
    /// and backtrack its chain. Returns every `(node, device)`
    /// commitment made (the caller merges them into its residency
    /// view). No-op for non-open rows.
    pub fn close_toward(
        &mut self,
        row_node: NodeId,
        consumer_device: DeviceLocation,
        edge_bytes: &[u64],
        est: &dyn TransferEstimator,
    ) -> Vec<(NodeId, DeviceLocation)> {
        let mut out = Vec::new();
        let Some(&idx) = self.by_node.get(&row_node) else {
            return out;
        };
        if self.rows[idx].status != RowStatus::Open {
            return out;
        }
        let row = &self.rows[idx];
        let mut choice: Option<(DeviceLocation, u64)> = None;
        for (j, &d) in row.devices.iter().enumerate() {
            let mut t = row.best[j];
            for &b in edge_bytes {
                t = t.saturating_add(est.estimate_transfer_ns(d, consumer_device, b));
            }
            if choice.is_none_or(|(_, c)| t < c) {
                choice = Some((d, t));
            }
        }
        if let Some((d, _)) = choice {
            self.commit_row(idx, d, &mut out);
        }
        out
    }

    /// Commit every remaining row: open rows price the exit
    /// crossing `transfer(d → exit_loc, exit_bytes(node))` (skipped
    /// when no realize-target location is known) and commit at the
    /// resulting argmin; their backpointers commit the chained rows
    /// behind them. Returns every commitment made.
    ///
    /// Defensive tail: a chained row not reachable from any open
    /// row's backtrack (which the chaining discipline should make
    /// impossible) commits at its own cheapest state rather than
    /// being left undecided — `compile_plan` errors typed if a row
    /// somehow has no committed device, never panics.
    pub fn finish(
        &mut self,
        exit_loc: Option<DeviceLocation>,
        exit_bytes: impl Fn(NodeId) -> u64,
        est: &dyn TransferEstimator,
    ) -> Vec<(NodeId, DeviceLocation)> {
        let mut out = Vec::new();
        for idx in 0..self.rows.len() {
            if self.rows[idx].status != RowStatus::Open {
                continue;
            }
            let row = &self.rows[idx];
            let mut choice: Option<(DeviceLocation, u64)> = None;
            for (j, &d) in row.devices.iter().enumerate() {
                let mut t = row.best[j];
                if let Some(loc) = exit_loc {
                    t = t.saturating_add(est.estimate_transfer_ns(
                        d,
                        loc,
                        exit_bytes(row.node),
                    ));
                }
                if choice.is_none_or(|(_, c)| t < c) {
                    choice = Some((d, t));
                }
            }
            if let Some((d, _)) = choice {
                self.commit_row(idx, d, &mut out);
            }
        }
        // Defensive: commit any row the backtracks missed.
        for idx in 0..self.rows.len() {
            if self.rows[idx].status == RowStatus::Committed {
                continue;
            }
            let row = &self.rows[idx];
            let mut choice: Option<(DeviceLocation, u64)> = None;
            for (j, &d) in row.devices.iter().enumerate() {
                if choice.is_none_or(|(_, c)| row.best[j] < c) {
                    choice = Some((d, row.best[j]));
                }
            }
            if let Some((d, _)) = choice {
                self.commit_row(idx, d, &mut out);
            }
        }
        out
    }

    /// Iterative backtrack (chains can be thousands of nodes —
    /// recursion would risk the stack): commit `start` at `device`,
    /// then follow its backpointers, committing each producer at
    /// the device the extension chose for this arrival device.
    fn commit_row(
        &mut self,
        start: usize,
        device: DeviceLocation,
        out: &mut Vec<(NodeId, DeviceLocation)>,
    ) {
        let mut stack = vec![(start, device)];
        while let Some((idx, dev)) = stack.pop() {
            if self.rows[idx].status == RowStatus::Committed {
                continue;
            }
            self.rows[idx].status = RowStatus::Committed;
            self.rows[idx].committed = Some(dev);
            out.push((self.rows[idx].node, dev));
            let Some(di) = self.rows[idx].devices.iter().position(|&d| d == dev)
            else {
                // Defensive: unknown arrival device → no backpointers
                // to follow; the finish() tail commits the chain.
                continue;
            };
            let back = self.rows[idx].back[di].clone();
            for (p_node, p_dev) in back {
                if let Some(&p_idx) = self.by_node.get(&p_node) {
                    stack.push((p_idx, p_dev));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    const CPU: DeviceLocation = DeviceLocation::Cpu;
    const GPU: DeviceLocation = DeviceLocation::Cuda { gpu_id: 0 };

    /// latency + bytes·ns_per_byte; zero same-device.
    struct Est {
        latency: u64,
        per_byte: u64,
    }

    impl TransferEstimator for Est {
        fn estimate_transfer_ns(
            &self,
            src: DeviceLocation,
            dst: DeviceLocation,
            bytes: u64,
        ) -> u64 {
            if src == dst {
                return 0;
            }
            self.latency.saturating_add(bytes.saturating_mul(self.per_byte))
        }
    }

    fn chain(producer: NodeId, bytes: u64) -> ChainInput {
        ChainInput { producer, edge_bytes: smallvec![bytes] }
    }

    #[test]
    fn extension_carries_accumulated_cost_and_backpointers() {
        let est = Est { latency: 1000, per_byte: 0 };
        let mut dp = PlacementDp::new();
        // n0: CPU 800, GPU 100; input fixed on CPU.
        dp.push_row(NodeId(0), &[(CPU, 800), (GPU, 100)], &[(CPU, 4)], &[], &est);
        assert!(dp.is_open(NodeId(0)));
        // n1 chains n0: CPU 800, GPU 100.
        dp.push_row(NodeId(1), &[(CPU, 800), (GPU, 100)], &[], &[chain(NodeId(0), 4)], &est);
        assert!(!dp.is_open(NodeId(0)), "chained producer leaves the frontier");
        assert!(dp.is_open(NodeId(1)));
        // best_n0 = [800, 1100]; best_n1[CPU] = 800 + min(800, 2100)
        // = 1600 (via CPU); best_n1[GPU] = 100 + min(1800, 1100) =
        // 1200 (via GPU).
        let commits = dp.finish(Some(CPU), |_| 4, &est);
        // Exit to CPU: CPU 1600 vs GPU 1200 + 1000 = 2200 → CPU,
        // backpointer commits n0 on CPU too.
        assert_eq!(dp.committed_device(NodeId(1)), Some(CPU));
        assert_eq!(dp.committed_device(NodeId(0)), Some(CPU));
        assert_eq!(commits.len(), 2);
    }

    #[test]
    fn exit_pricing_flips_the_local_winner() {
        let est = Est { latency: 1000, per_byte: 0 };
        let mut dp = PlacementDp::new();
        // Locally GPU wins: CPU 1050 vs GPU 100. Exit to CPU adds
        // 1000 to GPU → CPU wins.
        dp.push_row(NodeId(0), &[(CPU, 1050), (GPU, 100)], &[], &[], &est);
        dp.finish(Some(CPU), |_| 4, &est);
        assert_eq!(dp.committed_device(NodeId(0)), Some(CPU));

        // Without an exit location the local winner stands.
        let mut dp2 = PlacementDp::new();
        dp2.push_row(NodeId(0), &[(CPU, 1050), (GPU, 100)], &[], &[], &est);
        dp2.finish(None, |_| 4, &est);
        assert_eq!(dp2.committed_device(NodeId(0)), Some(GPU));
    }

    #[test]
    fn close_toward_prices_the_boundary_crossing() {
        let est = Est { latency: 10, per_byte: 1 };
        let mut dp = PlacementDp::new();
        // GPU locally cheaper (100 vs 300) but the consumer sits on
        // CPU and the boundary is 1024 bytes: GPU 100 + 1034 = 1134
        // vs CPU 300 → CPU.
        dp.push_row(NodeId(0), &[(CPU, 300), (GPU, 100)], &[], &[], &est);
        let commits = dp.close_toward(NodeId(0), CPU, &[1024], &est);
        assert_eq!(commits, vec![(NodeId(0), CPU)]);
        assert_eq!(dp.committed_device(NodeId(0)), Some(CPU));
        // Closing again is a no-op.
        assert!(dp.close_toward(NodeId(0), GPU, &[1024], &est).is_empty());
    }

    #[test]
    fn duplicated_edges_price_per_occurrence() {
        let est = Est { latency: 1000, per_byte: 0 };
        let mut dp = PlacementDp::new();
        dp.push_row(NodeId(0), &[(CPU, 0), (GPU, 0)], &[], &[], &est);
        // Add(x, x): two edges from the same producer.
        let ci = ChainInput { producer: NodeId(0), edge_bytes: smallvec![4, 4] };
        dp.push_row(NodeId(1), &[(CPU, 0), (GPU, 0)], &[], &[ci], &est);
        // Arriving GPU from a CPU producer pays two crossings
        // (2000); same-device pays zero, so ties keep everything on
        // the first-listed device after exit to CPU.
        dp.finish(Some(CPU), |_| 4, &est);
        assert_eq!(dp.committed_device(NodeId(1)), Some(CPU));
        assert_eq!(dp.committed_device(NodeId(0)), Some(CPU));
    }

    #[test]
    fn aliases_flatten_through_view_chains() {
        let mut dp = PlacementDp::new();
        let est = Est { latency: 0, per_byte: 0 };
        dp.push_row(NodeId(0), &[(CPU, 0)], &[], &[], &est);
        dp.add_alias(NodeId(1), NodeId(0));
        dp.add_alias(NodeId(2), NodeId(1));
        assert_eq!(dp.resolve(NodeId(2)), NodeId(0));
        assert_eq!(dp.resolve(NodeId(1)), NodeId(0));
        assert_eq!(dp.resolve(NodeId(3)), NodeId(3), "no alias → identity");
        assert!(dp.is_open(dp.resolve(NodeId(2))));
    }

    #[test]
    fn saturating_costs_never_overflow() {
        let est = Est { latency: u64::MAX, per_byte: u64::MAX };
        let mut dp = PlacementDp::new();
        dp.push_row(
            NodeId(0),
            &[(CPU, u64::MAX), (GPU, u64::MAX)],
            &[(CPU, u64::MAX), (GPU, u64::MAX)],
            &[],
            &est,
        );
        dp.push_row(
            NodeId(1),
            &[(CPU, u64::MAX)],
            &[],
            &[chain(NodeId(0), u64::MAX)],
            &est,
        );
        let commits = dp.finish(Some(GPU), |_| u64::MAX, &est);
        assert_eq!(commits.len(), 2, "all rows commit despite saturated costs");
    }
}

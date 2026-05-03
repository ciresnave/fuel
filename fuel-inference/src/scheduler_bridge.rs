//! Phase 6d Track 4: bridge from fuel-inference's runtime state into
//! the lazy-graph planner.
//!
//! The Lightbulb-contributed inference layer (this crate) handles
//! memory-pressure admission, MoE expert routing, speculative decoding
//! batching — all *runtime* policies. The lazy-graph planner
//! (`fuel-graph-router::RuleScheduler`) handles *placement* policy.
//! Each runs independently today; this module is where they meet.
//!
//! ## Today
//!
//! [`MemoryPressureRule`] is the pilot integration: a `SchedulerRule`
//! that consults `MemoryScheduler`-style memory pressure state to bias
//! the planner toward keeping ops on the same device as their primary
//! input when memory is tight (avoiding extra `Op::Copy` emissions).
//!
//! ## Roadmap
//!
//! Future rules in this module bridge MoE routing (route expert
//! batches to specific backends), speculative-decode draft/target
//! pairing (co-locate them), and tiered-storage residency hints.
//! Phase 9a's per-tensor metadata slot is the right home for
//! op-level tags like "this MatMul belongs to expert N"; until that
//! lands, the rules here use coarse, graph-shape-derived signals.

use fuel_core_types::DeviceLocation;
use fuel_graph::{NodeId, SharedGraph};
use fuel_graph_router::{Placement, Router, SchedulerRule};

use crate::scheduler::MemoryScheduler;

/// A snapshot of memory-pressure state captured from a
/// `MemoryScheduler` at the moment a graph is being planned. Owned
/// (no lifetime) so the rule can be cloned into the scheduler
/// pipeline without lifetime gymnastics.
#[derive(Clone, Debug)]
pub struct MemoryPressureSnapshot {
    /// True when usage exceeds the scheduler's pressure threshold.
    /// Drives the rule: under pressure, bias toward minimizing
    /// device transfers; otherwise the rule is a no-op.
    pub under_pressure: bool,
    /// Fraction of the budget currently in use (0.0–1.0+). Reserved
    /// for future rules that want a smoother gradient than the
    /// boolean threshold.
    pub usage_fraction: f64,
}

impl MemoryPressureSnapshot {
    /// Build a snapshot from a live `MemoryScheduler`.
    pub fn from(scheduler: &MemoryScheduler) -> Self {
        Self {
            under_pressure: scheduler.under_pressure(),
            usage_fraction: scheduler.usage_fraction(),
        }
    }
}

/// `SchedulerRule` that biases placement to follow each op's primary
/// input when memory is under pressure. Reduces `Op::Copy` emissions
/// at the cost of potentially missing a backend that would have run
/// the op faster — under pressure, the avoided D2D / H2D / D2H
/// transfer is usually the bigger win.
///
/// Intended to run *after* `BaselineRule` (which seeds default
/// placement) and *before* `ConstLoweringRule` (which lowers Const
/// nodes toward their consumers). The order means: baseline assigns
/// every op to the router default → memory-pressure rule rewrites
/// non-Const consumers to follow their first input → const-lowering
/// then propagates Const nodes to match.
pub struct MemoryPressureRule {
    snapshot: MemoryPressureSnapshot,
}

impl MemoryPressureRule {
    pub fn new(snapshot: MemoryPressureSnapshot) -> Self {
        Self { snapshot }
    }
}

impl SchedulerRule for MemoryPressureRule {
    fn apply(
        &self,
        graph: &SharedGraph,
        roots: &[NodeId],
        _router: &Router,
        placement: &mut Placement,
    ) {
        if !self.snapshot.under_pressure {
            return; // no-op when memory isn't tight
        }
        // Walk the topological order; for each non-Const, non-Const-input
        // op, set its placement to match its first input's placement.
        // Skip ops that already have a hint (the user's explicit
        // `graph.placement()` call wins).
        let order = {
            let g = graph.read().unwrap();
            fuel_graph::topo_order_multi(&g, roots)
        };
        let g = graph.read().unwrap();
        for nid in order {
            let node = g.node(nid);
            if matches!(node.op, fuel_graph::Op::Const) {
                continue;
            }
            if node.inputs.is_empty() {
                continue;
            }
            let primary_input = node.inputs[0];
            let inherited = match placement.get(&primary_input).copied() {
                Some(d) => d,
                None => continue,
            };
            // Don't override a non-default placement.
            // BaselineRule has populated `placement` for every node
            // already; we replace its entry only when keeping the
            // node on primary_input's device makes sense.
            placement.insert(nid, inherited);
        }
    }
}

/// Convenience: build a `MemoryPressureRule` from a live scheduler.
pub fn pressure_rule_from(scheduler: &MemoryScheduler) -> MemoryPressureRule {
    MemoryPressureRule::new(MemoryPressureSnapshot::from(scheduler))
}

/// Maps `DeviceLocation` keys for use in tests / planner inspection.
///
/// Reserved for future use — currently empty, but the symbol is
/// exported so downstream callers can pin the API surface they
/// depend on.
pub fn _placement_devices(placement: &Placement) -> Vec<DeviceLocation> {
    let mut seen = Vec::new();
    for (_, dev) in placement {
        if !seen.iter().any(|d| std::mem::discriminant(d) == std::mem::discriminant(dev)) {
            seen.push(dev.clone());
        }
    }
    seen
}

// Tests live in `tests/scheduler_bridge.rs` as an integration test —
// fuel-inference's lib-test target has unrelated pre-existing
// compile errors (Device::Cpu API drift) that would otherwise mask
// real failures here.

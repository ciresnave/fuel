//! Phase 6d Track 4: bridge from fuel-inference's runtime state into
//! graph placement planning.
//!
//! The Lightbulb-contributed inference layer (this crate) handles
//! memory-pressure admission, MoE expert routing, speculative decoding
//! batching — all *runtime* policies. Placement policy belongs to the
//! planner (today: the picker's `compile_plan` on the pipelined
//! executor; tomorrow: the load-time incremental planner). This module
//! is where the two meet.
//!
//! ## History
//!
//! The original bridge implemented `fuel-graph-router::SchedulerRule`
//! and plugged into the legacy `RuleScheduler` pipeline. That crate
//! retired with executor-unification Session 6 (2026-06-11); the rule
//! survives as a self-contained placement-bias pass over a plain
//! per-node placement map, with no legacy planner types in its
//! signature. The load-time planner program picks it up as a
//! placement-bias input when its admission/placement stages land.
//!
//! ## Today
//!
//! [`MemoryPressureRule`] is the pilot integration: a pass that
//! consults `MemoryScheduler`-style memory pressure state to bias
//! placement toward keeping ops on the same device as their primary
//! input when memory is tight (avoiding extra `Op::Copy` emissions).
//!
//! ## Roadmap
//!
//! Future rules in this module bridge MoE routing (route expert
//! batches to specific backends), speculative-decode draft/target
//! pairing (co-locate them), and tiered-storage residency hints.

use fuel_core_types::DeviceLocation;
use fuel_graph::{NodeId, SharedGraph};
use std::collections::HashMap;

use crate::scheduler::MemoryScheduler;

/// Per-node device assignment the bias passes read and refine.
/// Plain map — the planner-side consumer translates it into
/// `graph.set_placement` calls / plan pins as appropriate.
pub type Placement = HashMap<NodeId, DeviceLocation>;

/// A snapshot of memory-pressure state captured from a
/// `MemoryScheduler` at the moment a graph is being planned. Owned
/// (no lifetime) so the rule can be cloned into a planning
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

/// Placement-bias pass that makes each op follow its primary input's
/// device when memory is under pressure. Reduces `Op::Copy` emissions
/// at the cost of potentially missing a backend that would have run
/// the op faster — under pressure, the avoided D2D / H2D / D2H
/// transfer is usually the bigger win.
///
/// Intended to run over a placement map that has already been seeded
/// (every node assigned a baseline device): the pass rewrites
/// non-Const consumers to follow their first input.
pub struct MemoryPressureRule {
    snapshot: MemoryPressureSnapshot,
}

impl MemoryPressureRule {
    pub fn new(snapshot: MemoryPressureSnapshot) -> Self {
        Self { snapshot }
    }

    /// Apply the bias to `placement` for every node reachable from
    /// `roots`. No-op when the snapshot isn't under pressure.
    pub fn apply(&self, graph: &SharedGraph, roots: &[NodeId], placement: &mut Placement) {
        if !self.snapshot.under_pressure {
            return; // no-op when memory isn't tight
        }
        // Walk the topological order; for each non-Const op with
        // inputs, set its placement to match its first input's
        // placement.
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

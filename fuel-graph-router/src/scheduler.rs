//! Placement scheduler for multi-backend graphs.
//!
//! A [`Scheduler`] consumes a graph and produces a [`Placement`] —
//! a per-node assignment of `DeviceLocation`. The Router applies the
//! placement by calling [`fuel_graph::Graph::set_placement`] on every
//! assigned node, then runs [`fuel_graph::opt::lower_const_placement`]
//! and [`fuel_graph::opt::insert_copies`] to materialize the plan.
//!
//! ## Today: [`SimpleScheduler`]
//!
//! A trivial implementation that matches the existing Router behavior:
//! every node inherits the device of its first input (recursive), or
//! falls back to the router's default device. Used as the default so
//! existing callers get today's semantics with the new abstraction in
//! place.
//!
//! ## Tomorrow: [`RuleScheduler`]
//!
//! Planned rule-based scheduler (see project memory
//! `project_e_p4_p5_design_sketches.md`). Pipeline: baseline →
//! independent subgraphs → long-op migration → fusion-attract →
//! const-lowering → move-vs-recompute → redundant-move-elide. Each
//! rule is independently testable and near-linear. Ships in follow-up
//! sessions as the individual rules are implemented.

use fuel_core_types::DeviceLocation;
use fuel_graph::{topo_order_multi, NodeId, SharedGraph};
use std::collections::HashMap;

use crate::Router;

/// Per-node device assignment, the output of a [`Scheduler`].
///
/// A node without an entry in this map inherits its device via the
/// Router's default (today: follow the input's device; tomorrow: the
/// cost-minimizing choice).
pub type Placement = HashMap<NodeId, DeviceLocation>;

/// A placement scheduler. Runs before execution; its output drives
/// `graph.set_placement` for each assigned node, then
/// `opt::lower_const_placement` and `opt::insert_copies` materialize
/// the plan into a ready-to-realize graph.
pub trait Scheduler {
    /// Produce a placement for every node reachable from `roots`.
    /// The `router` is consulted for device capabilities and
    /// supported backends.
    fn plan(&self, graph: &SharedGraph, roots: &[NodeId], router: &Router) -> Placement;
}

/// Today-behavior scheduler: tag every node with the Router's default
/// device. Exists so downstream code can depend on the `Scheduler`
/// trait without committing to a specific scheduling strategy.
///
/// For Const nodes and nodes with multi-device inputs, the placement
/// this scheduler assigns gets refined by
/// [`fuel_graph::opt::lower_const_placement`] +
/// [`fuel_graph::opt::insert_copies`] in the downstream pipeline.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimpleScheduler;

impl Scheduler for SimpleScheduler {
    fn plan(&self, graph: &SharedGraph, roots: &[NodeId], router: &Router) -> Placement {
        let default_device = router.default_device();
        let g = graph.read().unwrap();
        let order = topo_order_multi(&g, roots);
        let mut out = Placement::with_capacity(order.len());
        for id in order {
            // Respect explicit graph-level placement hints; otherwise
            // assign the router's default.
            let d = g.placement(id).unwrap_or(default_device);
            out.insert(id, d);
        }
        out
    }
}

/// Apply a [`Placement`] to a graph by calling
/// `graph.set_placement(id, device)` for every assigned node.
/// Called by Router after [`Scheduler::plan`] to make the plan
/// visible to the subsequent `lower_const_placement` + `insert_copies`
/// passes.
pub fn apply_placement(graph: &SharedGraph, placement: &Placement) {
    let mut g = graph.write().unwrap();
    for (&id, &device) in placement {
        g.set_placement(id, device);
    }
}

// ---- Rule-based scheduler ---------------------------------------------------
//
// A rule-based scheduler runs a pipeline of small passes, each of which
// reads/writes the current Placement map. Each rule is independently
// testable and near-linear — matches how production compilers (TensorRT,
// XLA) structure their placement passes. See project memory
// `project_e_p4_p5_design_sketches.md` for the long-form design.

/// One stage of a [`RuleScheduler`] pipeline. Each rule reads the
/// current partial [`Placement`] and refines it.
pub trait SchedulerRule {
    fn apply(
        &self,
        graph: &SharedGraph,
        roots: &[NodeId],
        router: &Router,
        placement: &mut Placement,
    );
}

/// A graph-mutating rule stage. Runs AFTER every [`SchedulerRule`] in
/// the pipeline — by which time the placement map is fully populated
/// and the rule can read it to decide where emitted nodes should live.
///
/// Unlike [`SchedulerRule`], which only refines the placement map,
/// mutating rules can append nodes to the graph and rewrite consumer
/// input edges. They're the mechanism for residency eviction (emit
/// `Op::Copy` + `Op::Release` + reload chains), future fusion rules,
/// etc.
///
/// Ordering edges for any destructive ops the rule emits are derived
/// automatically by [`fuel_graph::opt::derive_ordering`] at realize
/// time; the rule does not need to express ordering itself.
pub trait GraphMutatingSchedulerRule {
    fn apply(
        &self,
        graph: &SharedGraph,
        roots: &[NodeId],
        router: &Router,
        placement: &mut Placement,
    );
}

/// Rule-based scheduler: runs a sequence of [`SchedulerRule`]s to
/// populate the [`Placement`] map, then a sequence of
/// [`GraphMutatingSchedulerRule`]s to apply graph surgery (residency
/// eviction, fusion, etc).
///
/// `default_pipeline()` gives a reasonable starter set for the
/// placement phase:
///   1. [`BaselineRule`] — every node gets router's default device,
///      overridden by any explicit `graph.placement()` hint.
///   2. [`ConstLoweringRule`] — Consts with unanimous consumer devices
///      get their placement lowered to that device, skipping a Copy
///      [`fuel_graph::opt::insert_copies`] would otherwise emit.
///
/// No mutating rules run by default. Callers opt in via
/// [`with_mutating_rule`][Self::with_mutating_rule] — typical: add a
/// residency-eviction rule parameterized by a VRAM budget.
pub struct RuleScheduler {
    rules: Vec<Box<dyn SchedulerRule>>,
    mutating_rules: Vec<Box<dyn GraphMutatingSchedulerRule>>,
}

impl RuleScheduler {
    pub fn new() -> Self {
        Self { rules: Vec::new(), mutating_rules: Vec::new() }
    }

    /// Append a placement-only rule to the pipeline.
    pub fn with_rule(mut self, rule: Box<dyn SchedulerRule>) -> Self {
        self.rules.push(rule);
        self
    }

    /// Append a graph-mutating rule. Runs after every non-mutating
    /// rule so the placement map is fully populated before the rule
    /// decides how to surgery the graph.
    pub fn with_mutating_rule(
        mut self,
        rule: Box<dyn GraphMutatingSchedulerRule>,
    ) -> Self {
        self.mutating_rules.push(rule);
        self
    }

    /// Baseline + const-lowering starter pipeline.
    pub fn default_pipeline() -> Self {
        Self::new()
            .with_rule(Box::new(BaselineRule))
            .with_rule(Box::new(ConstLoweringRule))
    }
}

impl Default for RuleScheduler {
    fn default() -> Self { Self::default_pipeline() }
}

impl Scheduler for RuleScheduler {
    fn plan(&self, graph: &SharedGraph, roots: &[NodeId], router: &Router) -> Placement {
        let mut placement = Placement::new();
        for rule in &self.rules {
            rule.apply(graph, roots, router, &mut placement);
        }
        for rule in &self.mutating_rules {
            rule.apply(graph, roots, router, &mut placement);
        }
        placement
    }
}

/// First-pass rule: assign every reachable node to the router's
/// default device, unless an explicit `graph.placement()` hint says
/// otherwise. Equivalent to [`SimpleScheduler`]'s entire behavior;
/// subsequent rules refine from this baseline.
pub struct BaselineRule;

impl SchedulerRule for BaselineRule {
    fn apply(
        &self,
        graph: &SharedGraph,
        roots: &[NodeId],
        router: &Router,
        placement: &mut Placement,
    ) {
        let default_device = router.default_device();
        let g = graph.read().unwrap();
        for id in fuel_graph::topo_order_multi(&g, roots) {
            let d = g.placement(id).unwrap_or(default_device);
            placement.insert(id, d);
        }
    }
}

/// Const-lowering rule: for each `Op::Const` node, if all its
/// consumers (per the current [`Placement`]) agree on a device, move
/// the const to that device. Saves a Move node that
/// [`fuel_graph::opt::insert_copies`] would otherwise emit.
///
/// Reads the per-node placement from `placement`, so [`BaselineRule`]
/// (or any rule that populates the map) must run first. Writes
/// refined Const placements back into `placement`.
pub struct ConstLoweringRule;

impl SchedulerRule for ConstLoweringRule {
    fn apply(
        &self,
        graph: &SharedGraph,
        roots: &[NodeId],
        _router: &Router,
        placement: &mut Placement,
    ) {
        // Build consumer index: for each node, the set of nodes that
        // consume it.
        let g = graph.read().unwrap();
        let order = fuel_graph::topo_order_multi(&g, roots);
        let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for &nid in &order {
            let node = g.node(nid);
            for &input in &node.inputs {
                consumers.entry(input).or_default().push(nid);
            }
        }

        // For each Const without an explicit graph-level hint, check
        // consumer unanimity in the current placement map.
        for &nid in &order {
            let is_const = matches!(g.node(nid).op, fuel_graph::Op::Const);
            if !is_const { continue; }
            if g.placement(nid).is_some() { continue; } // respect explicit hint

            let Some(cs) = consumers.get(&nid) else { continue };
            let mut target: Option<DeviceLocation> = None;
            let mut unanimous = true;
            for &c in cs {
                let d = match placement.get(&c) {
                    Some(&d) => d,
                    None => { unanimous = false; break; }
                };
                match target {
                    None => target = Some(d),
                    Some(prev) if prev == d => {}
                    Some(_) => { unanimous = false; break; }
                }
            }
            if unanimous {
                if let Some(d) = target {
                    placement.insert(nid, d);
                }
            }
        }
    }
}

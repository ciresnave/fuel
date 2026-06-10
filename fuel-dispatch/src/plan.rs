//! Execution plan — per-realize compilation output, Phase 1.5
//! reshape of the picker-work arc.
//!
//! The plan is a topological order plus a sparse `NodeId ->
//! AlternativeSet` map. Each entry is the per-decision-point top-N
//! candidates the optimizer ranker considered, filtered and ranked
//! per the configured `PlanOptions`. The eventual executor reads
//! `alternatives` per node, takes the winner (Picker 2 territory in
//! the long-term shape; Phase 4 wires the executor to this surface),
//! and dispatches the resolved kernel.
//!
//! ## What changed in Phase 1.5
//!
//! Pre-1.5 this module hosted `NodeKernelBinding`, `TolerancePolicy`,
//! and `resolve_kernel` — the v1 picker that fell out of Phase 7.6
//! step 9b. Those types had zero callers (verified by the
//! 2026-05-30 picker-alternatives audit), so Phase 1.5 retires
//! them and ships the AlternativeSet-based plan in their place. The
//! shape composes with everything Phase 1.1–1.4 already shipped:
//! the enumerator builds candidates, the filter chain narrows them,
//! the cost composer scores them, and the plan stores the
//! top-N-after-filter-and-rank.
//!
//! Phase 4 of the picker-work arc migrates the PipelinedExecutor to
//! consume `ExecutionPlan::alternatives` instead of `compile_node`'s
//! first-registered path. Until then this module ships the
//! infrastructure with no production consumer.

use std::collections::HashMap;

use fuel_core_types::dispatch::{OpKind, SizeClass};
use fuel_core_types::probe::BackendId;
use fuel_core_types::{DType, DeviceLocation, Error, Result, Shape};
use fuel_graph::{Graph, NodeId};

use crate::kernel::KernelBindingTable;
use crate::pipelined::{build_lookup_dtypes, op_to_op_kind};
use crate::ranker::{
    apply_filter_chain, compute_static_costs, default_chain, enumerate_candidates,
    AlternativeSet, CapabilitiesLookup, DecisionContext, FilterContext, JudgeOracle,
    PrecisionRequirement, DEFAULT_MAX_N,
};

/// Per-realize execution plan. Built by [`compile_plan`].
///
/// `alternatives` is sparse — view ops, `Op::Const`, and ops
/// `op_to_op_kind` returns `None` for get no entry. The executor's
/// per-node dispatch reads `alternatives.get(&id)` and either takes
/// the winner (top entry after rank) or, in the long-term shape,
/// hands the full set to the runtime selector (Picker 2) for
/// layer-3-telemetry-driven choice.
#[derive(Debug)]
pub struct ExecutionPlan {
    /// Topological order — same shape the executor walks.
    pub order: Vec<NodeId>,
    /// One `AlternativeSet` per kernel-bearing node. Sparse.
    pub alternatives: HashMap<NodeId, AlternativeSet>,
    /// `SystemTopology` generation counter snapshotted at
    /// [`compile_plan`] time. The executor (Phase 4.3) checks this
    /// against `crate::dispatch::topology_generation()` at every
    /// dispatch-chunk boundary; mismatch surfaces
    /// [`Error::TopologyChanged`] so the realize layer can rebuild
    /// the plan against the fresh topology.
    ///
    /// `0` for [`ExecutionPlan::empty()`] — empty plans by definition
    /// have no alternatives to invalidate, so the executor's chunk-
    /// boundary check skips them.
    pub generation: u64,
}

impl ExecutionPlan {
    /// Empty plan. Tests + future direct-call sites.
    pub fn empty() -> Self {
        Self {
            order: Vec::new(),
            alternatives: HashMap::new(),
            generation: 0,
        }
    }

    /// Read-only handle to a node's alternative set. `None` for
    /// nodes outside the plan's map.
    pub fn alternatives(&self, node: NodeId) -> Option<&AlternativeSet> {
        self.alternatives.get(&node)
    }

    /// Mutable handle — used by post-plan refinement passes (Phase
    /// 3's Judge integration may re-rank top-N after the plan is
    /// built).
    pub fn alternatives_mut(&mut self, node: NodeId) -> Option<&mut AlternativeSet> {
        self.alternatives.get_mut(&node)
    }
}

/// Default [`DeviceLocation`] for a `BackendId` when the graph has
/// no per-node placement set. Mirrors the convention in
/// `fuel-graph-router` (CPU is the singleton `Cpu`; GPU backends
/// default to ordinal 0).
pub fn default_device_for(backend: BackendId) -> DeviceLocation {
    match backend {
        BackendId::Cpu => DeviceLocation::Cpu,
        BackendId::Cuda => DeviceLocation::Cuda { gpu_id: 0 },
        BackendId::Vulkan => DeviceLocation::Vulkan { gpu_id: 0 },
        BackendId::Metal => DeviceLocation::Metal { gpu_id: 0 },
        _ => DeviceLocation::Cpu,
    }
}

/// The `BackendId` that owns a `DeviceLocation`'s storage substrate.
/// Total — `BackendId` mirrors `DeviceLocation` 1:1 (AOCL/MKL are
/// `kernel_source` siblings under `Cpu`, not distinct backends).
/// Inverse of [`default_device_for`] modulo GPU ordinals.
pub fn backend_for_device(loc: DeviceLocation) -> BackendId {
    match loc {
        DeviceLocation::Cpu => BackendId::Cpu,
        DeviceLocation::Cuda { .. } => BackendId::Cuda,
        DeviceLocation::Vulkan { .. } => BackendId::Vulkan,
        DeviceLocation::Metal { .. } => BackendId::Metal,
    }
}

/// Plan-time configuration the caller hands to [`compile_plan`].
///
/// Most fields have sensible defaults via [`PlanOptions::new`] +
/// chained builders. The two callback fields are required for
/// non-trivial usage — the `'env` lifetime ties them to the
/// caller's environment.
///
/// # `placements_for_device`
///
/// When `Some`, called per kernel-bearing node with the node's
/// target `DeviceLocation` and expected to return every
/// `BackendId` the picker is allowed to consider at that device.
/// This is the cross-co-located-backend integration point —
/// callers wire it to `SystemTopology::backends_for(device)` to
/// unlock the AOCL/MKL/CPU competition story.
///
/// When `None`, the planner uses single-backend placement: only
/// `(target_backend, placement)` from the graph node's side-table.
/// Matches the pre-1.5 picker's pinned-backend behavior; useful as
/// a transitional default and for tests.
///
/// # `pinned_device`
///
/// Picker-arc step 4a. When `Some`, nodes WITHOUT an explicit graph
/// placement resolve their decision device to this location — the
/// realize call's pinned DEVICE. This replaces the bridge's
/// pre-plan monolithic per-node `set_target_backend` loop: the
/// graph no longer needs `target_backend` stamped before planning;
/// the plan's per-node winner is stamped back onto the graph AFTER
/// planning (`fuel-core::pipelined_bridge::stamp_plan_backends`).
///
/// Resolution priority per node: explicit `Graph::placement` (the
/// scheduler's per-node assignment) → `pinned_device` →
/// `default_device_for(target_backend)` (legacy stamped graphs) →
/// error.
///
/// # `fallback_placements_for`
///
/// Picker-arc step 4b. When the node's decision device has NO
/// registered implementation for its `(op, dtypes)` — primary
/// enumeration came back empty — this closure supplies the
/// OFF-DEVICE placements the picker may consider instead, making
/// the CPU-fallback case a first-class plan-time decision instead
/// of an executor-level special case. Locality is preserved: the
/// closure is never consulted while the decision device has at
/// least one registered candidate.
///
/// Constraints enforced by `compile_plan`:
///
/// - **Destructive ops never fall back** (`Op::destructive_input()`
///   is `Some`): in-place mutation semantics don't survive moving
///   the op away from the device that owns its mutation target;
///   those ops keep the plan-time `NoBackendForOp` error.
/// - **Fallback sets freeze to their single ranked winner**: the
///   residency stitch (`Op::Copy` insertion) is a graph rewrite
///   computed from the static winner, so a dispatch-time selector
///   must not be able to pick a sibling on a different device.
///
/// # `capabilities_for`
///
/// Required when `populate_costs` is true (the default). The
/// closure resolves a `BackendId` to its `BackendCapabilities` —
/// callers typically wire it to `SystemTopology::capabilities(...)`
/// or the CapabilityRegistry. Returns `None` for backends not in
/// the topology (their candidates retain default zero cost).
pub struct PlanOptions<'env> {
    /// Hard precision floor the picker enforces. Default: empty
    /// (unconstrained — every candidate passes the precision filter).
    pub precision_requirement: PrecisionRequirement,
    /// Top-N retention per decision point. Default:
    /// [`DEFAULT_MAX_N`] (3) per architecture v1.0 §04.
    pub max_alternatives_per_node: usize,
    /// Whether to invoke [`compute_static_costs`] + rank after
    /// filtering. Default: true. Disable for tests that just want
    /// to verify enumeration + filtering without cost machinery.
    pub populate_costs: bool,
    /// Cross-co-located-backend placement enumerator. See struct
    /// docs.
    pub placements_for_device:
        Option<&'env (dyn Fn(DeviceLocation) -> Vec<BackendId> + 'env)>,
    /// Realize-call device pin (picker-arc step 4a). See struct
    /// docs.
    pub pinned_device: Option<DeviceLocation>,
    /// Off-device fallback enumerator for missing-impl ops
    /// (picker-arc step 4b). See struct docs.
    pub fallback_placements_for: Option<
        &'env (dyn Fn(DeviceLocation) -> Vec<(BackendId, DeviceLocation)> + 'env),
    >,
    /// Capabilities lookup. Required when `populate_costs` is
    /// true.
    pub capabilities_for: Option<&'env CapabilitiesLookup<'env>>,
    /// Optional Phase 3 Layer-2 oracle. When `Some` AND
    /// `populate_costs` is true, the cost composer refines each
    /// candidate's Layer-1 estimate with the Judge's measured
    /// latency at that cell. Candidates the oracle hasn't measured
    /// keep the Layer-1 estimate (silent fallback — absence ≠ zero).
    pub judge: Option<&'env dyn JudgeOracle>,
}

impl Default for PlanOptions<'_> {
    fn default() -> Self {
        Self {
            precision_requirement: PrecisionRequirement::default(),
            max_alternatives_per_node: DEFAULT_MAX_N,
            populate_costs: true,
            placements_for_device: None,
            pinned_device: None,
            fallback_placements_for: None,
            capabilities_for: None,
            judge: None,
        }
    }
}

impl<'env> PlanOptions<'env> {
    /// Builder constructor; same fields as [`Default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the precision requirement hard filter.
    pub fn with_precision_requirement(mut self, req: PrecisionRequirement) -> Self {
        self.precision_requirement = req;
        self
    }

    /// Set the top-N alternative retention bound.
    pub fn with_max_alternatives(mut self, n: usize) -> Self {
        self.max_alternatives_per_node = n;
        self
    }

    /// Disable cost computation + ranking. Sets `populate_costs =
    /// false`; alternatives end up in enumerator order, useful for
    /// tests that don't care about ranking.
    pub fn without_cost_population(mut self) -> Self {
        self.populate_costs = false;
        self
    }

    /// Attach a co-located-backend enumerator. Wire to
    /// `SystemTopology::backends_for(device)` for production.
    pub fn with_placements_for_device(
        mut self,
        f: &'env (dyn Fn(DeviceLocation) -> Vec<BackendId> + 'env),
    ) -> Self {
        self.placements_for_device = Some(f);
        self
    }

    /// Pin the realize call's device. Nodes without an explicit
    /// graph placement enumerate their candidates at this location
    /// — no per-node `target_backend` stamp needed before planning.
    /// Picker-arc step 4a.
    pub fn with_pinned_device(mut self, device: DeviceLocation) -> Self {
        self.pinned_device = Some(device);
        self
    }

    /// Attach an off-device fallback enumerator, consulted ONLY
    /// when a node's decision device has no registered
    /// implementation for its `(op, dtypes)`. Wire to the
    /// `SystemTopology` device list for production. Picker-arc
    /// step 4b — see struct docs for the destructive-op and
    /// single-winner constraints.
    pub fn with_fallback_placements_for(
        mut self,
        f: &'env (dyn Fn(DeviceLocation) -> Vec<(BackendId, DeviceLocation)> + 'env),
    ) -> Self {
        self.fallback_placements_for = Some(f);
        self
    }

    /// Attach a capabilities lookup. Required when `populate_costs`
    /// is true.
    pub fn with_capabilities_for(mut self, f: &'env CapabilitiesLookup<'env>) -> Self {
        self.capabilities_for = Some(f);
        self
    }

    /// Attach a Layer-2 Judge oracle. When set, `compile_plan`'s
    /// cost composer refines Layer-1 estimates with measured
    /// latencies per [`JudgeOracle::measured_latency_ns`].
    pub fn with_judge(mut self, judge: &'env dyn JudgeOracle) -> Self {
        self.judge = Some(judge);
        self
    }
}

/// Build an [`ExecutionPlan`] from a topologically-ordered node
/// sequence + binding table + options.
///
/// For each node in `order`:
///
/// - If `op_to_op_kind(&node.op)` returns `None` (view ops,
///   `Op::Const`, ops not yet wired into dispatch), the node gets
///   no entry — view-only adoption + const adoption flow through
///   the legacy paths unchanged.
/// - `Op::Copy` nodes get no entry either (picker-arc step 4a):
///   their kernel backend is residency-determined (the SOURCE
///   backend), so transfer-kernel resolution stays with the
///   executor's `compile_node` path keyed by the bridge-maintained
///   source-backend stamp.
/// - Otherwise:
///   1. Resolve `(op_kind, dtypes, device)` from the graph + the
///      options (`device` via `Graph::placement`, then
///      `PlanOptions::pinned_device`, then
///      `default_device_for(target_backend)` for legacy stamped
///      graphs).
///   2. Build the placement list — either via
///      `options.placements_for_device` (cross-backend mode) or
///      `[(target_backend, device)]` (legacy single-backend mode).
///   3. Enumerate candidates via [`enumerate_candidates`].
///   4. Apply the default filter chain (precision floor +
///      strided-input pref + bit-stable pref).
///   5. If `populate_costs`, compute static costs via the
///      capabilities lookup and rank ascending by composite cost.
///   6. Truncate to `max_alternatives_per_node`.
///   7. If the set is empty after filtering, return
///      [`Error::NoBackendForOp`] (fail-fast at plan time, not
///      deep in executor dispatch).
///   8. Insert into `plan.alternatives`.
pub fn compile_plan(
    graph: &Graph,
    order: &[NodeId],
    bindings_table: &KernelBindingTable,
    options: &PlanOptions<'_>,
) -> Result<ExecutionPlan> {
    // Snapshot the topology generation at plan-build time. The
    // executor (Phase 4.3) checks this against the live counter at
    // every dispatch-chunk boundary; mismatch surfaces
    // `Error::TopologyChanged` so the realize layer can rebuild
    // against the fresh topology.
    let generation = crate::dispatch::topology_generation();
    let mut alternatives_map = HashMap::with_capacity(order.len());

    for &id in order {
        let node = graph.node(id);
        let Some(op_kind) = op_to_op_kind(&node.op) else {
            continue;
        };

        // Op::Copy is residency-determined, not a picker decision:
        // its kernel runs on the backend that owns the SOURCE bytes
        // (`copy_from_cpu_wrapper` for H2D, the source backend's
        // download wrapper for D2H). Enumerating it against a single
        // decision device would key the lookup at the wrong end of
        // the transfer for placement-carrying copies (the consumer
        // device, where the H2D copy's kernel does NOT run). The
        // executor's legacy `compile_node` path resolves these via
        // the source-backend `target_backend` stamp maintained by
        // the bridge's copy-insertion passes. Picker-arc step 4a.
        if matches!(node.op, fuel_graph::Op::Copy { .. }) {
            continue;
        }

        // Decision-device resolution (step 4a): explicit per-node
        // placement (scheduler assignments) → the realize call's
        // pinned device → the legacy stamped-backend default.
        let explicit_backend = graph.target_backend(id);
        let target_device = graph
            .placement(id)
            .or(options.pinned_device)
            .or_else(|| explicit_backend.map(default_device_for))
            .ok_or_else(|| {
                Error::Msg(format!(
                    "compile_plan: node {:?} ({:?}) has no device context — \
                     set PlanOptions::pinned_device, a graph placement, or \
                     the node's target_backend",
                    id, node.op,
                ))
                .bt()
            })?;
        // Diagnostic backend for error reporting; also the legacy
        // single-backend placement when no topology enumerator is
        // configured.
        let diag_backend =
            explicit_backend.unwrap_or_else(|| backend_for_device(target_device));
        let dtypes = build_lookup_dtypes(graph, node);

        // Build the placement list: cross-backend if a topology
        // enumerator is configured, otherwise single-backend
        // legacy mode.
        let placements: Vec<(BackendId, DeviceLocation)> =
            match options.placements_for_device {
                Some(f) => f(target_device)
                    .into_iter()
                    .map(|backend| (backend, target_device))
                    .collect(),
                None => vec![(diag_backend, target_device)],
            };

        let op_params = candidate_default_op_params(graph, node);
        let mut set = enumerate_candidates(
            op_kind,
            &dtypes,
            &placements,
            &op_params,
            bindings_table,
            options.max_alternatives_per_node,
        );

        // Picker-arc step 4b: when the decision device has NO
        // implementation for this (op, dtypes), admit off-device
        // candidates from the fallback enumerator. The missing-impl
        // op becomes a plan-time picker decision (the bridge
        // stitches residency via Op::Copy insertion around the
        // off-device winner) instead of a realize-time error.
        // Destructive ops never fall back — in-place mutation
        // semantics don't survive moving the op away from the
        // device that owns its mutation target.
        let mut from_fallback = false;
        if set.is_empty() && node.op.destructive_input().is_none() {
            if let Some(fallback) = options.fallback_placements_for {
                let fb_placements = fallback(target_device);
                if !fb_placements.is_empty() {
                    set = enumerate_candidates(
                        op_kind,
                        &dtypes,
                        &fb_placements,
                        &op_params,
                        bindings_table,
                        options.max_alternatives_per_node,
                    );
                    from_fallback = !set.is_empty();
                }
            }
        }

        // Fail-fast: if enumeration found nothing (on the decision
        // device AND via fallback), surface the missing-binding
        // error before filters can also empty the set (which would
        // produce a less-specific FilterRejected).
        if set.is_empty() {
            return Err(missing_binding_error(
                bindings_table,
                op_kind,
                &dtypes,
                diag_backend,
            ));
        }

        // Stamp the decision-point identity so dispatch-time
        // selectors (Picker 2) can re-query the Judge per candidate.
        // The derivation mirrors `compute_static_costs`'s Layer-2
        // lookup key exactly: principal dtype = first lookup dtype,
        // size class = first input's element count (SizeClass(0)
        // for nullary ops).
        if let Some(&principal_dtype) = dtypes.first() {
            let size_class = node
                .inputs
                .first()
                .map(|&input_id| {
                    SizeClass::from_elem_count(graph.node(input_id).shape.elem_count())
                })
                .unwrap_or(SizeClass(0));
            set.set_context(DecisionContext {
                op: op_kind,
                principal_dtype,
                size_class,
            });
        }

        // Apply the default filter chain.
        let input_layouts: Vec<fuel_core_types::Layout> = node
            .inputs
            .iter()
            .map(|&input_id| graph.layout(input_id))
            .collect();
        let ctx = FilterContext::new(op_kind, &dtypes, &input_layouts);
        let chain = default_chain(options.precision_requirement);
        apply_filter_chain(&mut set, &chain, &ctx)?;

        // Cost composition + rank (optional — tests may skip).
        if options.populate_costs {
            if let Some(caps_for) = options.capabilities_for {
                let input_shapes: Vec<Shape> = node
                    .inputs
                    .iter()
                    .map(|&input_id| graph.node(input_id).shape.clone())
                    .collect();
                compute_static_costs(
                    &mut set,
                    op_kind,
                    &dtypes,
                    &input_shapes,
                    bindings_table,
                    caps_for,
                    options.judge,
                );
                set.rank_by_composite_cost();
            }
        }

        set.truncate_to_top_n();

        // Step 4b: off-device fallback sets freeze to their single
        // ranked winner. The bridge's residency stitch (Op::Copy
        // insertion) is a graph rewrite computed from the static
        // winner; leaving siblings on OTHER devices in the set
        // would let a dispatch-time selector pick a candidate whose
        // inputs were never copied to its device.
        if from_fallback && set.len() > 1 {
            set.retain_indices(&[0]);
        }

        // After truncation an empty set is the surfaceable error
        // (the filter chain or rank dropped everything). The chain
        // would have raised FilterRejected on hard-empty, so this
        // path is only reachable when populate_costs + ranking
        // pruned a structurally-empty set, which shouldn't happen
        // in practice — defensive only.
        if set.is_empty() {
            return Err(missing_binding_error(
                bindings_table,
                op_kind,
                &dtypes,
                diag_backend,
            ));
        }

        alternatives_map.insert(id, set);
    }

    Ok(ExecutionPlan {
        order: order.to_vec(),
        alternatives: alternatives_map,
        generation,
    })
}

/// Build an `Error::NoBackendForOp` diagnostic for a decision
/// point with no registered alternative. Same shape as
/// [`crate::kernel::KernelBindingTable::lookup_with_caps`]'s error
/// branch so callers see consistent output regardless of which
/// path surfaced the miss.
fn missing_binding_error(
    table: &KernelBindingTable,
    op: OpKind,
    dtypes: &[DType],
    backend: BackendId,
) -> Error {
    let _ = (table, backend);
    Error::NoBackendForOp {
        op,
        dtypes: dtypes.to_vec(),
        available_backends: Vec::new(),
        supported_combinations: Vec::new(),
    }
    .bt()
}

/// Helper that produces the `OpParams` to attach to enumerated
/// candidates for one graph node. Phase 1.5 ships with the
/// placeholder `OpParams::None` for every node — the live
/// op-params shape derivation lives in `pipelined::op_to_op_params`
/// but is currently mid-refactor for other reasons. Phase 4's
/// executor integration replaces this with the live derivation so
/// the planner's candidate set matches what dispatch sees.
fn candidate_default_op_params(
    _graph: &Graph,
    _node: &fuel_graph::Node,
) -> crate::kernel::OpParams {
    crate::kernel::OpParams::None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::PrecisionGuarantee;
    use crate::kernel::{unknown_cost, KernelCaps, OpParams};
    use fuel_core_types::backend::{
        BackendCapabilities, SubstrateClass, TransferPath,
    };
    use fuel_core_types::{DType, Layout, Result as FuelResult, Shape, StrideVec};
    use fuel_graph::{topo_order, Node, Op};
    use fuel_storage::Storage;
    use smallvec::smallvec;
    use std::collections::HashSet;
    use std::sync::{Arc, RwLock};

    fn noop_kernel(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> FuelResult<()> {
        Ok(())
    }

    fn noop_kernel_b(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> FuelResult<()> {
        Ok(())
    }

    fn register_add_f32(
        table: &mut KernelBindingTable,
        backend: BackendId,
        kernel: crate::kernel::KernelRef,
        precision: PrecisionGuarantee,
    ) {
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            backend,
            kernel,
            KernelCaps::empty(),
            precision,
            unknown_cost,
        );
    }

    fn build_add_graph() -> (Graph, NodeId) {
        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let rhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let add = g.push(Node {
            op: Op::Add,
            inputs: vec![lhs, rhs],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        g.set_target_backend(add, BackendId::Cpu);
        (g, add)
    }

    fn cpu_caps() -> BackendCapabilities {
        BackendCapabilities {
            backend_id: BackendId::Cpu,
            device_location: DeviceLocation::Cpu,
            op_dtype_support: HashSet::new(),
            required_alignment: 64,
            access_granularity_bits: 8,
            transfer_paths: vec![(DeviceLocation::Cpu, TransferPath::SameDevice)],
            storage_substrate: SubstrateClass::HostBytes,
        }
    }

    #[test]
    fn empty_plan_has_no_alternatives() {
        let p = ExecutionPlan::empty();
        assert!(p.order.is_empty());
        assert!(p.alternatives.is_empty());
        assert!(p.alternatives(NodeId(0)).is_none());
    }

    #[test]
    fn default_device_per_backend_matches_router_convention() {
        assert_eq!(default_device_for(BackendId::Cpu), DeviceLocation::Cpu);
        assert_eq!(
            default_device_for(BackendId::Cuda),
            DeviceLocation::Cuda { gpu_id: 0 },
        );
        assert_eq!(
            default_device_for(BackendId::Vulkan),
            DeviceLocation::Vulkan { gpu_id: 0 },
        );
    }

    #[test]
    fn compile_plan_skips_view_and_const_nodes() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );

        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[6]),
            dtype: DType::F32,
        });
        let rhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[6]),
            dtype: DType::F32,
        });
        let add = g.push(Node {
            op: Op::Add,
            inputs: vec![lhs, rhs],
            shape: Shape::from_dims(&[6]),
            dtype: DType::F32,
        });
        let reshape = g.push(Node {
            op: Op::Reshape(Shape::from_dims(&[2, 3])),
            inputs: vec![add],
            shape: Shape::from_dims(&[2, 3]),
            dtype: DType::F32,
        });
        g.set_target_backend(add, BackendId::Cpu);
        g.set_target_backend(reshape, BackendId::Cpu);

        let order = topo_order(&g, reshape);
        let opts = PlanOptions::new().without_cost_population();
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");

        assert_eq!(plan.order.len(), 4);
        assert_eq!(
            plan.alternatives.len(),
            1,
            "only Add carries an alternative set",
        );
        let alts = plan.alternatives(add).expect("Add alternatives present");
        assert_eq!(alts.len(), 1);
        assert!(plan.alternatives(lhs).is_none());
        assert!(plan.alternatives(rhs).is_none());
        assert!(plan.alternatives(reshape).is_none());
    }

    #[test]
    fn compile_plan_fails_fast_on_missing_binding() {
        let table = KernelBindingTable::new();
        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[2, 3]),
            dtype: DType::F32,
        });
        let rhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3, 2]),
            dtype: DType::F32,
        });
        let mm = g.push(Node {
            op: Op::MatMul,
            inputs: vec![lhs, rhs],
            shape: Shape::from_dims(&[2, 2]),
            dtype: DType::F32,
        });
        g.set_target_backend(mm, BackendId::Cpu);

        let order = topo_order(&g, mm);
        let opts = PlanOptions::new().without_cost_population();
        let err = compile_plan(&g, &order, &table, &opts).unwrap_err();
        match err {
            Error::NoBackendForOp { op, dtypes, .. } => {
                assert_eq!(op, OpKind::MatMul);
                assert_eq!(dtypes, vec![DType::F32, DType::F32, DType::F32]);
            }
            other => panic!("expected NoBackendForOp, got {other:?}"),
        }
    }

    #[test]
    fn compile_plan_legacy_single_backend_uses_target_backend() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        // Even with an AOCL alternative registered, the legacy mode
        // (placements_for_device = None) only uses the node's
        // target_backend (Cpu).
        register_add_f32(
            &mut table,
            BackendId::Cuda,
            noop_kernel_b,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let opts = PlanOptions::new().without_cost_population();
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add_id).unwrap();
        assert_eq!(alts.len(), 1, "legacy mode → single backend → single alt");
        assert_eq!(alts.winner().unwrap().backend, BackendId::Cpu);
    }

    #[test]
    fn compile_plan_cross_co_located_via_placements_callback() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        register_add_f32(
            &mut table,
            BackendId::Cuda,
            noop_kernel_b,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );

        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu {
                vec![BackendId::Cpu, BackendId::Cuda]
            } else {
                vec![]
            }
        };
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_placements_for_device(&placements_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add_id).unwrap();
        assert_eq!(
            alts.len(),
            2,
            "cross-co-located mode aggregates Cpu + Aocl alternatives",
        );
    }

    #[test]
    fn compile_plan_precision_requirement_filters_via_default_chain() {
        let mut table = KernelBindingTable::new();
        // Non-bit-stable.
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee {
                bit_stable_on_same_hardware: false,
                ..PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU
            },
        );

        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_precision_requirement(PrecisionRequirement::BIT_STABLE);
        let err = compile_plan(&g, &order, &table, &opts).unwrap_err();
        match err {
            Error::FilterRejected { filter, .. } => {
                assert_eq!(filter, "precision-floor");
            }
            other => panic!("expected FilterRejected, got {other:?}"),
        }
        // Sanity: drop the requirement → plan succeeds.
        let unconstrained = PlanOptions::new().without_cost_population();
        let plan = compile_plan(&g, &order, &table, &unconstrained).expect("compile");
        assert!(plan.alternatives(add_id).is_some());
        let _ = add_id;
    }

    #[test]
    fn compile_plan_with_cost_ranks_cheaper_winner() {
        fn cheap_cost(
            _: &[Shape],
            _: &[DType],
            _: &OpParams,
            _: &BackendCapabilities,
        ) -> crate::fused::CostEstimate {
            crate::fused::CostEstimate { flops: 10, bytes_moved: 0, kernel_overhead_ns: 0 }
        }
        fn expensive_cost(
            _: &[Shape],
            _: &[DType],
            _: &OpParams,
            _: &BackendCapabilities,
        ) -> crate::fused::CostEstimate {
            crate::fused::CostEstimate {
                flops: 1_000_000,
                bytes_moved: 0,
                kernel_overhead_ns: 0,
            }
        }
        let mut table = KernelBindingTable::new();
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            noop_kernel,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            expensive_cost,
        );
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cuda,
            noop_kernel_b,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            cheap_cost,
        );

        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu {
                vec![BackendId::Cpu, BackendId::Cuda]
            } else {
                vec![]
            }
        };
        let cpu_caps_val = cpu_caps();
        let caps_fn = |_: BackendId| Some(&cpu_caps_val);
        let opts = PlanOptions::new()
            .with_placements_for_device(&placements_fn)
            .with_capabilities_for(&caps_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add_id).unwrap();
        assert_eq!(alts.len(), 2);
        assert_eq!(
            alts.winner().unwrap().backend,
            BackendId::Cuda,
            "cheaper static cost wins after ranking",
        );
    }

    #[test]
    fn compile_plan_judge_layer2_can_flip_layer1_winner() {
        // Layer-1 says CPU is cheap, Aocl is expensive.
        // Layer-2 (Judge) measured the opposite: Aocl 20 ns, CPU 5000 ns.
        // After compile_plan's cost composition + rank, Aocl wins.
        use crate::ranker::HashMapJudge;
        use fuel_core_types::dispatch::SizeClass;

        fn cpu_layer1(
            _: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities,
        ) -> crate::fused::CostEstimate {
            crate::fused::CostEstimate { flops: 10, bytes_moved: 0, kernel_overhead_ns: 0 }
        }
        fn aocl_layer1(
            _: &[Shape], _: &[DType], _: &OpParams, _: &BackendCapabilities,
        ) -> crate::fused::CostEstimate {
            crate::fused::CostEstimate { flops: 100_000, bytes_moved: 0, kernel_overhead_ns: 0 }
        }
        let mut table = KernelBindingTable::new();
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu, noop_kernel, KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU, cpu_layer1,
        );
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cuda, noop_kernel_b, KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU, aocl_layer1,
        );

        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu {
                vec![BackendId::Cpu, BackendId::Cuda]
            } else { vec![] }
        };
        let cpu_caps_val = cpu_caps();
        let caps_fn = |_: BackendId| Some(&cpu_caps_val);

        // Each input is shape [3] (per build_add_graph); the
        // first-input-shape rule gives elem_count 3.
        let sc = SizeClass::from_elem_count(3);
        let mut judge = HashMapJudge::new();
        judge.insert(OpKind::AddElementwise, DType::F32, sc, BackendId::Cpu, "", 5_000);
        judge.insert(OpKind::AddElementwise, DType::F32, sc, BackendId::Cuda, "", 20);

        let opts = PlanOptions::new()
            .with_placements_for_device(&placements_fn)
            .with_capabilities_for(&caps_fn)
            .with_judge(&judge);

        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add_id).unwrap();
        assert_eq!(alts.len(), 2);
        assert_eq!(
            alts.winner().unwrap().backend,
            BackendId::Cuda,
            "Layer-2 measurement reverses Layer-1 verdict via compile_plan",
        );
    }

    /// `compile_plan` stamps each set with the decision-point
    /// context using the same key derivation as the cost composer's
    /// Layer-2 lookup: principal dtype = first lookup dtype, size
    /// class = first input's element count.
    #[test]
    fn compile_plan_stamps_decision_context() {
        use fuel_core_types::dispatch::SizeClass;

        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let opts = PlanOptions::new().without_cost_population();
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add_id).expect("Add alternatives present");
        let ctx = alts.context().expect("context stamped");
        assert_eq!(ctx.op, OpKind::AddElementwise);
        assert_eq!(ctx.principal_dtype, DType::F32);
        // build_add_graph inputs are shape [3].
        assert_eq!(ctx.size_class, SizeClass::from_elem_count(3));
    }

    #[test]
    fn compile_plan_truncates_to_max_n() {
        let mut table = KernelBindingTable::new();
        // Three CPU-substrate backends competing at one decision
        // point — `truncate_to_top_n` should keep only the top 2.
        for backend in [BackendId::Cpu, BackendId::Cuda, BackendId::Vulkan] {
            register_add_f32(
                &mut table,
                backend,
                if backend == BackendId::Cpu { noop_kernel } else { noop_kernel_b },
                PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            );
        }

        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu {
                vec![
                    BackendId::Cpu,
                    BackendId::Cuda,
                    BackendId::Vulkan,
                ]
            } else {
                vec![]
            }
        };
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_max_alternatives(2)
            .with_placements_for_device(&placements_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add_id).unwrap();
        assert_eq!(alts.len(), 2, "truncated to max_alternatives_per_node");
    }

    /// Step 4a parity: an UNSTAMPED graph planned with
    /// `with_pinned_device(Cpu)` produces the same alternative sets
    /// as the legacy stamped graph — candidate-for-candidate
    /// (backend, device, kernel, kernel_source).
    #[test]
    fn compile_plan_pinned_device_matches_stamped_plan() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );

        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu {
                vec![BackendId::Cpu]
            } else {
                vec![]
            }
        };

        // Legacy: stamped target_backend, no pinned device.
        let (g_stamped, add_stamped) = build_add_graph();
        let order_stamped = topo_order(&g_stamped, add_stamped);
        let opts_stamped = PlanOptions::new()
            .without_cost_population()
            .with_placements_for_device(&placements_fn);
        let plan_stamped =
            compile_plan(&g_stamped, &order_stamped, &table, &opts_stamped)
                .expect("stamped compile");

        // New: same graph shape, NO target_backend anywhere, pinned
        // device instead.
        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let rhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let add = g.push(Node {
            op: Op::Add,
            inputs: vec![lhs, rhs],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let order = topo_order(&g, add);
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_placements_for_device(&placements_fn)
            .with_pinned_device(DeviceLocation::Cpu);
        let plan = compile_plan(&g, &order, &table, &opts).expect("pinned compile");

        let a = plan_stamped.alternatives(add_stamped).expect("stamped set");
        let b = plan.alternatives(add).expect("pinned set");
        assert_eq!(a.len(), b.len(), "same candidate count");
        for (ca, cb) in a.alternatives().iter().zip(b.alternatives()) {
            assert_eq!(ca.backend, cb.backend);
            assert_eq!(ca.device, cb.device);
            assert_eq!(ca.kernel as usize, cb.kernel as usize, "same kernel ref");
            assert_eq!(ca.kernel_source, cb.kernel_source);
        }
        assert_eq!(a.context(), b.context(), "same decision context");
    }

    /// Step 4a: a node with neither a placement, nor a pinned
    /// device, nor a target_backend stamp is a plan-time error —
    /// fail-fast with an actionable message.
    #[test]
    fn compile_plan_errors_without_any_device_context() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let add = g.push(Node {
            op: Op::Add,
            inputs: vec![lhs, lhs],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let order = topo_order(&g, add);
        let opts = PlanOptions::new().without_cost_population();
        let err = compile_plan(&g, &order, &table, &opts).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("no device context"),
            "expected the device-context diagnostic, got: {msg}",
        );
    }

    /// Step 4a: `Op::Copy` nodes are excluded from the plan map —
    /// their kernel is residency-determined (source backend), so
    /// dispatch flows through the executor's legacy `compile_node`
    /// path keyed by the bridge's source-backend stamp.
    #[test]
    fn compile_plan_skips_op_copy_nodes() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        // A Copy registration exists — the skip must not depend on
        // the table lacking one.
        table.register_full(
            OpKind::Copy,
            &[DType::F32, DType::F32],
            BackendId::Cpu,
            noop_kernel_b,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            unknown_cost,
        );

        let (mut g, add_id) = build_add_graph();
        let copy_id = g.push(Node {
            op: Op::Copy { target: DeviceLocation::Cpu },
            inputs: vec![add_id],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        g.set_target_backend(copy_id, BackendId::Cpu);
        let order = topo_order(&g, copy_id);
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_pinned_device(DeviceLocation::Cpu);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        assert!(plan.alternatives(add_id).is_some(), "Add is planned");
        assert!(
            plan.alternatives(copy_id).is_none(),
            "Op::Copy is residency-determined — no plan entry",
        );
    }

    /// Step 4b: an op with no implementation on the pinned device
    /// falls back to an off-device candidate — the missing-impl op
    /// becomes a plan-time picker decision.
    #[test]
    fn compile_plan_fallback_admits_off_device_candidate() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let mut table = KernelBindingTable::new();
        // Add f32 registered ONLY on Cpu — the pinned CUDA device
        // has no impl.
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );

        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let add = g.push(Node {
            op: Op::Add,
            inputs: vec![lhs, lhs],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let order = topo_order(&g, add);

        let placements_fn = move |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == cuda0 { vec![BackendId::Cuda] } else { vec![BackendId::Cpu] }
        };
        let fallback_fn = |_dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            vec![(BackendId::Cpu, DeviceLocation::Cpu)]
        };
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_pinned_device(cuda0)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add).expect("Add planned via fallback");
        assert_eq!(alts.len(), 1, "fallback set frozen to one winner");
        let w = alts.winner().unwrap();
        assert_eq!(w.backend, BackendId::Cpu);
        assert_eq!(w.device, DeviceLocation::Cpu, "off-device placement");
    }

    /// Step 4b locality: the fallback enumerator is NOT consulted
    /// while the pinned device has a registered candidate — even if
    /// the off-device sibling would rank cheaper.
    #[test]
    fn compile_plan_fallback_not_consulted_when_pinned_has_impl() {
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        register_add_f32(
            &mut table,
            BackendId::Cuda,
            noop_kernel_b,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let placements_fn = |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == DeviceLocation::Cpu { vec![BackendId::Cpu] } else { vec![] }
        };
        let fallback_fn = |_dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            panic!(
                "fallback consulted although the pinned device has an \
                 implementation — locality policy violated",
            );
        };
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_pinned_device(DeviceLocation::Cpu)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add_id).unwrap();
        assert_eq!(alts.len(), 1);
        assert_eq!(alts.winner().unwrap().device, DeviceLocation::Cpu);
    }

    /// Step 4b: destructive ops never fall back — the plan-time
    /// NoBackendForOp error is preserved.
    #[test]
    fn compile_plan_fallback_denied_for_destructive_ops() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let mut table = KernelBindingTable::new();
        // ReluInplace registered ONLY on Cpu; pinned device is CUDA.
        table.register_full(
            OpKind::ReluInplace,
            &[DType::F32, DType::F32],
            BackendId::Cpu,
            noop_kernel,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            unknown_cost,
        );
        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let relu = g.push(Node {
            op: Op::ReluInplace,
            inputs: vec![lhs],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let order = topo_order(&g, relu);
        let placements_fn = move |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == cuda0 { vec![BackendId::Cuda] } else { vec![BackendId::Cpu] }
        };
        let fallback_fn = |_dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            vec![(BackendId::Cpu, DeviceLocation::Cpu)]
        };
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_pinned_device(cuda0)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn);
        let err = compile_plan(&g, &order, &table, &opts).unwrap_err();
        assert!(
            matches!(err, Error::NoBackendForOp { op: OpKind::ReluInplace, .. }),
            "destructive op must NOT fall back off-device; got {err:?}",
        );
    }

    /// Step 4b: even when MULTIPLE off-device backends could serve
    /// the op, the fallback set freezes to a single winner so the
    /// dispatch-time selector can't diverge from the residency
    /// stitch.
    #[test]
    fn compile_plan_fallback_freezes_to_single_winner() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let vk0 = DeviceLocation::Vulkan { gpu_id: 0 };
        let mut table = KernelBindingTable::new();
        register_add_f32(
            &mut table,
            BackendId::Cpu,
            noop_kernel,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );
        register_add_f32(
            &mut table,
            BackendId::Vulkan,
            noop_kernel_b,
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
        );

        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let add = g.push(Node {
            op: Op::Add,
            inputs: vec![lhs, lhs],
            shape: Shape::from_dims(&[3]),
            dtype: DType::F32,
        });
        let order = topo_order(&g, add);
        let placements_fn = move |dev: DeviceLocation| -> Vec<BackendId> {
            if dev == cuda0 { vec![BackendId::Cuda] } else { vec![] }
        };
        let fallback_fn = move |_dev: DeviceLocation| -> Vec<(BackendId, DeviceLocation)> {
            vec![
                (BackendId::Cpu, DeviceLocation::Cpu),
                (BackendId::Vulkan, vk0),
            ]
        };
        let opts = PlanOptions::new()
            .without_cost_population()
            .with_pinned_device(cuda0)
            .with_placements_for_device(&placements_fn)
            .with_fallback_placements_for(&fallback_fn);
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add).unwrap();
        assert_eq!(
            alts.len(),
            1,
            "fallback set must freeze to its single ranked winner",
        );
    }

    #[test]
    fn compile_plan_passes_input_layouts_to_filter_chain() {
        // Smoke test: a non-contiguous input + strided-input pref
        // in the default chain prefers the strided-capable kernel.
        // Register two CPU kernels, one with strided caps.
        let mut table = KernelBindingTable::new();
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            noop_kernel,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            unknown_cost,
        );
        table.register_full(
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            noop_kernel_b,
            KernelCaps::strided_input(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            unknown_cost,
        );

        // Build a graph where the Add's LHS input is non-contiguous.
        let mut g = Graph::new();
        let lhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[6]),
            dtype: DType::F32,
        });
        // Force a non-contiguous layout on lhs by setting a custom
        // stride via the graph's layout side-table.
        let strides: StrideVec = smallvec![2isize];
        g.set_layout(lhs, Layout::new(Shape::from_dims(&[6]), strides, 0));
        let rhs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[6]),
            dtype: DType::F32,
        });
        let add = g.push(Node {
            op: Op::Add,
            inputs: vec![lhs, rhs],
            shape: Shape::from_dims(&[6]),
            dtype: DType::F32,
        });
        g.set_target_backend(add, BackendId::Cpu);
        let order = topo_order(&g, add);
        let opts = PlanOptions::new().without_cost_population();
        let plan = compile_plan(&g, &order, &table, &opts).expect("compile");
        let alts = plan.alternatives(add).unwrap();
        // Strided-input preference narrowed to the strided-capable
        // kernel.
        assert_eq!(alts.len(), 1);
        assert!(alts.winner().unwrap().caps.strided_input);
    }
}

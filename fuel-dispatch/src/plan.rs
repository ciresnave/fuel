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
/// - Otherwise:
///   1. Resolve `(op_kind, dtypes, target_backend, device)` from
///      the graph (`target_backend` via `Graph::target_backend`,
///      `device` via `Graph::placement` or `default_device_for`).
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

        let target_backend = graph.target_backend(id).ok_or_else(|| {
            Error::Msg(format!(
                "compile_plan: node {:?} ({:?}) has no target_backend set",
                id, node.op,
            ))
            .bt()
        })?;
        let target_device = graph
            .placement(id)
            .unwrap_or_else(|| default_device_for(target_backend));
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
                None => vec![(target_backend, target_device)],
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

        // Fail-fast: if enumeration found nothing, surface the
        // missing-binding error before filters can also empty the
        // set (which would produce a less-specific
        // FilterRejected).
        if set.is_empty() {
            return Err(missing_binding_error(
                bindings_table,
                op_kind,
                &dtypes,
                target_backend,
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
                target_backend,
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

//! Execution plan + lazy kernel binding. Phase 7.6 step 9b.
//!
//! [`ExecutionPlan`] is the per-realize compilation output: a
//! topological order plus a sparse `NodeId -> NodeKernelBinding` map.
//! Each binding names a `(OpKind, dtypes, BackendId, DeviceLocation)`
//! decision point and starts with `kernel: None`.
//!
//! [`NodeKernelBinding::kernel`] is filled lazily by the route picker
//! ([`resolve_kernel`], step A3) the first time the executor reaches
//! the node — `lookup_alternatives` returns the registered siblings,
//! the picker chooses one under a [`TolerancePolicy`], and the chosen
//! `KernelRef` is cached on the binding for the rest of the realize.
//!
//! ## What lives where
//!
//! - **Types** ([`NodeKernelBinding`], [`ExecutionPlan`]): in this
//!   step (A1) — the shape every later step (A2/A3/A4 + Track B's
//!   executor migration) populates and reads.
//! - **`compile_plan`** (step A2): walks the graph, builds bindings
//!   for every kernel-bearing node, and fail-fast-asserts that the
//!   binding-table has at least one alternative for each
//!   `(op_kind, dtypes, backend)` triple. View ops, `Op::Const`, and
//!   ops the binding-table doesn't index (yet) get no binding.
//! - **`resolve_kernel`** + [`TolerancePolicy`] (step A3): the v1
//!   route picker. `BitStableFirst` (default) prefers alternatives
//!   with `bit_stable_on_same_hardware: true`; `FirstAlternative`
//!   exposes registration-order for tests.
//!
//! ## Out of scope (per session prompt)
//!
//! - Empirical Judge integration (replaces the v1 picker in a later
//!   phase).
//! - Real `KernelRevisionHash` computation (stays
//!   [`KernelRevisionHash::UNTRACKED`] until the persistence-cache
//!   phase starts).
//! - Optimizer-level decomposition-vs-fused alternatives at the same
//!   decision point (phase 7.6 step 10 or later).
//! - `Op::Fused` arms (the binding-table indexes primitives; fused
//!   ops route through [`crate::fused::FusedKernelRegistry`] — a
//!   parallel path that Track B's `compile_plan_fused_node` will
//!   handle).

use std::collections::HashMap;

use fuel_core_types::dispatch::OpKind;
use fuel_core_types::probe::BackendId;
use fuel_core_types::{DType, DeviceLocation, Error, Result};
use fuel_graph::{Graph, NodeId};
use smallvec::SmallVec;

use crate::fused::KernelRevisionHash;
use crate::kernel::{KernelBindingTable, KernelDTypes, KernelRef};
use crate::pipelined::{build_lookup_dtypes, op_to_op_kind};

/// One node's lazy kernel resolution.
///
/// Per architecture v1.0 §04 ("Per-decision-point alternatives"): each
/// kernel-bearing node carries a binding that names its decision point
/// (`op_kind`, `dtypes`, `backend`, `device`) and a `kernel` slot that
/// starts `None`. The route picker fills the slot at first use; the
/// chosen [`KernelRef`] is cached for the rest of the realize so a
/// single decision point resolves once per realize regardless of how
/// many times the executor reaches the node.
///
/// `kernel_revision` is recorded for cache-invalidation by the future
/// persistence layer ([`docs/architecture/11-persistence.md`]). Today
/// it stays [`KernelRevisionHash::UNTRACKED`] — real hashing lands
/// alongside the cache work.
#[derive(Clone, Debug)]
pub struct NodeKernelBinding {
    /// The graph node this binding describes.
    pub node: NodeId,
    /// The op-kind family the binding-table indexes the kernel under.
    pub op_kind: OpKind,
    /// Per-operand dtype list (inputs in order, then outputs) — the
    /// same shape as the binding-table's key.
    pub dtypes: KernelDTypes,
    /// Backend the kernel will run on. Driven by the Router
    /// (`Graph::target_backend(id)`) at plan time.
    pub backend: BackendId,
    /// Specific device within `backend` (multi-GPU CUDA carries the
    /// GPU ordinal; CPU is the singleton `Cpu` value). Derived from
    /// `Graph::placement(id)` when set, otherwise the backend's
    /// default device.
    pub device: DeviceLocation,
    /// The resolved kernel, or `None` until the route picker has run.
    /// Lazy by construction — see module doc.
    pub kernel: Option<KernelRef>,
    /// Revision of the resolved kernel; the persistence layer uses
    /// this to detect kernel drift across cache load. Stays
    /// [`KernelRevisionHash::UNTRACKED`] in step 9b.
    pub kernel_revision: KernelRevisionHash,
}

/// Per-realize execution plan. Built once by `compile_plan` (step A2)
/// at the start of a realize call; consumed by the executor (Track B)
/// or by tests (Track A).
///
/// The plan is the substrate the route picker writes into. The
/// executor's per-node dispatch reads `bindings.get(&node_id)`,
/// resolves the kernel via [`resolve_kernel`] (step A3), and invokes
/// the cached `KernelRef`. Nodes without a binding (view ops,
/// `Op::Const`, ops the binding-table doesn't index yet) flow through
/// the legacy paths unchanged.
///
/// ### Storage choice — HashMap vs Vec
///
/// `bindings` is a `HashMap` because the mapping is sparse:
/// view ops, `Op::Const`, and yet-to-migrate ops have no entry, so a
/// `Vec<Option<NodeKernelBinding>>` indexed by topo position would be
/// mostly empty. If profiling shows the realize hot path eats
/// HashMap lookups once Track B's executor migration lands, revisit
/// and consider a `Vec<NodeKernelBinding>` keyed by an internal
/// per-plan index plus a `NodeId -> usize` translation map.
#[derive(Debug)]
pub struct ExecutionPlan {
    /// Topological order — same shape the executor walks today.
    /// `compile_plan` clones the caller's order rather than recomputing
    /// it; the prevailing pattern in fuel-graph-executor is to compute
    /// order once via `execution_plan(&graph, &roots)` and then pass
    /// it down.
    pub order: Vec<NodeId>,
    /// One binding per kernel-bearing node in `order`. Sparse — see
    /// the doc-comment above.
    pub bindings: HashMap<NodeId, NodeKernelBinding>,
}

impl ExecutionPlan {
    /// Empty plan (no nodes, no bindings). Used by tests; production
    /// callers go through `compile_plan` (step A2).
    pub fn empty() -> Self {
        Self {
            order: Vec::new(),
            bindings: HashMap::new(),
        }
    }

    /// Convenience: a binding's mutable handle, used by
    /// [`resolve_kernel`] to cache the chosen `KernelRef` on first
    /// resolution. Returns `None` if the node has no binding — view
    /// ops, `Op::Const`, etc.
    pub fn binding_mut(&mut self, node: NodeId) -> Option<&mut NodeKernelBinding> {
        self.bindings.get_mut(&node)
    }

    /// Read-only handle to a node's binding. `None` for nodes outside
    /// the plan's binding map.
    pub fn binding(&self, node: NodeId) -> Option<&NodeKernelBinding> {
        self.bindings.get(&node)
    }
}

/// Default [`DeviceLocation`] for a `BackendId` when the graph has no
/// per-node placement set. Mirrors the convention in
/// `fuel-graph-router` (CPU is the singleton `Cpu`; GPU backends
/// default to ordinal 0).
pub(crate) fn default_device_for(backend: BackendId) -> DeviceLocation {
    match backend {
        BackendId::Reference
        | BackendId::Cpu
        | BackendId::Aocl
        | BackendId::Mkl => DeviceLocation::Cpu,
        BackendId::Cuda => DeviceLocation::Cuda { gpu_id: 0 },
        BackendId::Vulkan => DeviceLocation::Vulkan { gpu_id: 0 },
        BackendId::Metal => DeviceLocation::Metal { gpu_id: 0 },
        // BackendId is `#[non_exhaustive]`: future backends default
        // to CPU's singleton placeholder until they wire their own
        // arm here.
        _ => DeviceLocation::Cpu,
    }
}

/// Build an [`ExecutionPlan`] from a topologically-ordered node
/// sequence and the binding-table snapshot. Step A2 of Phase 7.6
/// step 9b.
///
/// For every node in `order`:
///
/// - If `op_to_op_kind(&node.op)` returns `None` (view ops,
///   `Op::Const`, ops not yet wired into the dispatch table),
///   the node gets **no binding** — view-only adoption and const
///   adoption flow through the legacy paths unchanged.
/// - Otherwise resolve the node's `(op_kind, dtypes, backend, device)`
///   tuple — `dtypes` via [`build_lookup_dtypes`] (same shape the
///   pipelined path uses), `backend` via `Graph::target_backend(id)`
///   (populated by op-builder methods per Phase 7.5 B3),
///   `device` via `Graph::placement(id)` (falling back to
///   `default_device_for(backend)`).
/// - **Fail-fast guard**: assert the binding-table has at least one
///   alternative registered for `(op_kind, dtypes, backend)`. If none,
///   return [`Error::NoBackendForOp`] — failing at plan time beats
///   failing at first-use time deep inside `eval_node`. Resolution
///   (*which* alternative wins) is lazy; *existence* is checked
///   eagerly.
/// - Insert `NodeKernelBinding { kernel: None, kernel_revision: UNTRACKED, ... }`
///   into `plan.bindings`. The route picker (step A3) fills `kernel`
///   on first use.
///
/// `order` is the same topological order today's executor walks — the
/// pipelined path computes it via `fuel_graph::topo_order`; callers
/// pass it in rather than recomputing here.
pub fn compile_plan(
    graph: &Graph,
    order: &[NodeId],
    bindings_table: &KernelBindingTable,
) -> Result<ExecutionPlan> {
    let mut bindings = HashMap::with_capacity(order.len());

    for &id in order {
        let node = graph.node(id);
        // Skip ops the binding-table doesn't index: view ops, Const,
        // Reshape, and any op_to_op_kind() returns None for.
        let Some(op_kind) = op_to_op_kind(&node.op) else {
            continue;
        };
        let backend = graph.target_backend(id).ok_or_else(|| {
            Error::Msg(format!(
                "compile_plan: node {:?} ({:?}) has no target_backend set",
                id, node.op,
            ))
            .bt()
        })?;
        let device = graph
            .placement(id)
            .unwrap_or_else(|| default_device_for(backend));
        let dtypes = build_lookup_dtypes(graph, node);

        // Fail-fast: the binding-table must carry at least one
        // alternative for this decision point. Missing-binding errors
        // surface here, not deep in eval_node where the executor
        // would otherwise hit them.
        let alts = bindings_table.lookup_alternatives(op_kind, &dtypes, backend);
        if alts.is_empty() {
            // Reuse the error shape lookup_with_caps emits so callers
            // see consistent diagnostics whether they go through the
            // legacy path or compile_plan.
            return Err(missing_binding_error(
                bindings_table,
                op_kind,
                &dtypes,
                backend,
            ));
        }

        bindings.insert(
            id,
            NodeKernelBinding {
                node: id,
                op_kind,
                dtypes: SmallVec::from_slice(&dtypes),
                backend,
                device,
                kernel: None,
                kernel_revision: KernelRevisionHash::UNTRACKED,
            },
        );
    }

    Ok(ExecutionPlan {
        order: order.to_vec(),
        bindings,
    })
}

/// Build an [`Error::NoBackendForOp`] diagnostic for a decision point
/// with no registered alternative. Same shape as
/// [`crate::kernel::KernelBindingTable::lookup_with_caps`]'s error
/// branch so callers see consistent output regardless of which path
/// surfaced the miss.
fn missing_binding_error(
    table: &KernelBindingTable,
    op: OpKind,
    dtypes: &[DType],
    backend: BackendId,
) -> Error {
    let _ = (table, backend); // table-introspection is the legacy path's job;
    // here we surface only the op + dtypes the caller asked for. A
    // richer "available backends for this op" enumeration could be
    // added once compile_plan starts being the primary error surface
    // (Track B onward).
    Error::NoBackendForOp {
        op,
        dtypes: dtypes.to_vec(),
        available_backends: Vec::new(),
        supported_combinations: Vec::new(),
    }
    .bt()
}

/// v1 route picker (step A3 of Phase 7.6 step 9b). Resolves a
/// [`NodeKernelBinding`]'s `kernel` slot from the alternatives the
/// binding-table registers at its decision point, caches the chosen
/// [`KernelRef`] for subsequent calls, and returns it.
///
/// The v1 picker is intentionally a placeholder: per architecture
/// v1.0 §04, the long-term home for selection is the empirical Judge
/// driven by per-cell telemetry — out of scope for 9b. Today's choice
/// is a discrete tolerance policy ([`TolerancePolicy`]); the Judge
/// integration is the future replacement.
///
/// ## Semantics
///
/// - **First-call:** the binding's `kernel` is `None`. The picker
///   reads `bindings_table.lookup_alternatives(...)`, applies
///   `policy`, writes the result back to `binding.kernel`, and
///   returns it.
/// - **Second-call:** `binding.kernel` is `Some(_)` from the prior
///   resolution. The picker short-circuits — no table lookup — and
///   returns the cached value. This is the lazy-caching commitment
///   architecture v1.0 §04 names: a decision point resolves once per
///   realize.
/// - **No alternative registered:** returns [`Error::NoBackendForOp`].
///   In a well-formed plan this never fires — `compile_plan` (A2)
///   already verified ≥1 alternative existed. The branch exists so
///   the picker is safe to call against a binding from outside
///   `compile_plan` (e.g. tests, future direct-call sites).
///
/// `kernel_revision` will be updated alongside `kernel` once
/// per-kernel revision hashing exists (out of scope for 9b — stays
/// [`KernelRevisionHash::UNTRACKED`]).
pub fn resolve_kernel(
    binding: &mut NodeKernelBinding,
    bindings_table: &KernelBindingTable,
    policy: TolerancePolicy,
) -> Result<KernelRef> {
    if let Some(k) = binding.kernel {
        return Ok(k);
    }
    let alts = bindings_table.lookup_alternatives(
        binding.op_kind,
        &binding.dtypes,
        binding.backend,
    );
    let chosen = match policy {
        TolerancePolicy::BitStableFirst => alts
            .iter()
            .find(|e| e.precision.bit_stable_on_same_hardware)
            .or_else(|| alts.first()),
        TolerancePolicy::FirstAlternative => alts.first(),
    }
    .ok_or_else(|| {
        Error::NoBackendForOp {
            op: binding.op_kind,
            dtypes: binding.dtypes.to_vec(),
            available_backends: Vec::new(),
            supported_combinations: Vec::new(),
        }
        .bt()
    })?;
    binding.kernel = Some(chosen.kernel);
    // kernel_revision stays UNTRACKED in 9b — real revision hashing
    // lands with the persistence-cache phase. Keeping the field on
    // the binding ensures the seam is in place when that work starts.
    Ok(chosen.kernel)
}

/// Route-picker tolerance policy. The v1 picker (step 9b) exposes
/// two arms; architecture v1.0 §04's long-term shape is a per-op
/// tolerance *budget* (`max_ulp_threshold: u32`, `max_relative_threshold: f64`)
/// driven by calibration data — that's the future replacement, not 9b
/// scope.
///
/// The discrete enum is a placeholder for two reasons:
///
/// 1. **Calibration framework isn't built yet.** Without measured
///    per-cell error data, there's nothing to drive a budget-based
///    picker — every alternative would look "good enough."
///    `BitStableFirst` is the conservative stand-in: prefer
///    bit-equivalent kernels until measured data argues otherwise.
/// 2. **The cutlass session's architectural payoff.** Registering
///    CUTLASS as a bf16/f16 matmul sibling alongside cuBLAS doesn't
///    change user behavior under `BitStableFirst` (cuBLAS wins on
///    `bit_stable_on_same_hardware: true`). A later session enables
///    CUTLASS by flipping a tolerance-policy switch — no executor
///    edit required.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TolerancePolicy {
    /// **Default.** Prefer an alternative whose
    /// `PrecisionGuarantee::bit_stable_on_same_hardware` is `true`. If
    /// none of the registered alternatives are bit-stable, fall back
    /// to the first-registered alternative.
    BitStableFirst,
    /// First-registered alternative wins, regardless of precision.
    /// Used by tests that exercise non-bit-stable kernels deliberately,
    /// and by future entry points (a "tolerance-aware" realize call)
    /// that have already filtered alternatives by their per-op
    /// tolerance budget upstream.
    FirstAlternative,
}

impl Default for TolerancePolicy {
    /// `BitStableFirst` — architecture v1.0 §04 names bit-stability as
    /// the default correctness anchor; the picker honors that until a
    /// calibration framework can drive richer policy.
    fn default() -> Self {
        TolerancePolicy::BitStableFirst
    }
}

/// Build a [`NodeKernelBinding`] with `kernel: None` — the
/// pre-resolution shape every entry in `ExecutionPlan::bindings`
/// starts as. Exposed for step A4's tests; production construction
/// flows through `compile_plan` (A2).
pub(crate) fn empty_binding(
    node: NodeId,
    op_kind: OpKind,
    dtypes: &[DType],
    backend: BackendId,
    device: DeviceLocation,
) -> NodeKernelBinding {
    NodeKernelBinding {
        node,
        op_kind,
        dtypes: SmallVec::from_slice(dtypes),
        backend,
        device,
        kernel: None,
        kernel_revision: KernelRevisionHash::UNTRACKED,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, RwLock};

    use fuel_core_types::{Layout, Shape};
    use fuel_graph::{topo_order, Node, Op};

    use crate::fused::PrecisionGuarantee;
    use crate::kernel::{KernelCaps, OpParams};
    use crate::kernel::unknown_cost;
    use crate::Storage;

    // ---------- A1 type-shape tests (kept from earlier commit) ----------

    #[test]
    fn empty_plan_has_no_bindings() {
        let plan = ExecutionPlan::empty();
        assert!(plan.order.is_empty());
        assert!(plan.bindings.is_empty());
    }

    #[test]
    fn default_device_per_backend_matches_router_convention() {
        assert_eq!(default_device_for(BackendId::Reference), DeviceLocation::Cpu);
        assert_eq!(default_device_for(BackendId::Cpu), DeviceLocation::Cpu);
        assert_eq!(default_device_for(BackendId::Aocl), DeviceLocation::Cpu);
        assert_eq!(default_device_for(BackendId::Mkl), DeviceLocation::Cpu);
        assert_eq!(
            default_device_for(BackendId::Cuda),
            DeviceLocation::Cuda { gpu_id: 0 },
        );
        assert_eq!(
            default_device_for(BackendId::Vulkan),
            DeviceLocation::Vulkan { gpu_id: 0 },
        );
        assert_eq!(
            default_device_for(BackendId::Metal),
            DeviceLocation::Metal { gpu_id: 0 },
        );
    }

    #[test]
    fn empty_binding_starts_with_kernel_none_and_untracked_revision() {
        let binding = empty_binding(
            NodeId(0),
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            DeviceLocation::Cpu,
        );
        assert_eq!(binding.node, NodeId(0));
        assert_eq!(binding.op_kind, OpKind::AddElementwise);
        assert_eq!(
            binding.dtypes.as_slice(),
            &[DType::F32, DType::F32, DType::F32],
        );
        assert_eq!(binding.backend, BackendId::Cpu);
        assert_eq!(binding.device, DeviceLocation::Cpu);
        assert!(binding.kernel.is_none());
        assert_eq!(binding.kernel_revision, KernelRevisionHash::UNTRACKED);
    }

    // ---------- A4 compile_plan + resolve_kernel tests ----------

    /// No-op kernel stand-ins. Distinct `fn` items so the binding-
    /// table's append-on-register treats them as sibling alternatives
    /// (registering the *same* fn item twice is the panic-guarded
    /// programmer-error path; we want two real alternatives here).
    fn ok_kernel_a(
        _inputs: &[Arc<RwLock<Storage>>],
        _outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        _params: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn ok_kernel_b(
        _inputs: &[Arc<RwLock<Storage>>],
        _outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        _params: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    /// PrecisionGuarantee carrying `bit_stable_on_same_hardware: true`
    /// without further bound claims — used to mark one alternative as
    /// the bit-stable choice in the BitStableFirst-policy tests.
    const BIT_STABLE: PrecisionGuarantee = PrecisionGuarantee {
        bit_stable_on_same_hardware: true,
        max_ulp: Some(0),
        max_relative: None,
        max_absolute: None,
        notes: "test bit-stable stub",
    };

    /// Helper: register one binding-table entry with explicit
    /// precision (so we can pick which alternative carries
    /// bit_stable_on_same_hardware = true). Defaults to KernelCaps::empty()
    /// and unknown_cost — tests don't exercise caps or cost.
    fn register(
        table: &mut KernelBindingTable,
        op: OpKind,
        dtypes: &[DType],
        backend: BackendId,
        kernel: crate::kernel::KernelRef,
        precision: PrecisionGuarantee,
    ) {
        table.register_full(
            op,
            dtypes,
            backend,
            kernel,
            KernelCaps::empty(),
            precision,
            unknown_cost,
        );
    }

    /// Build a 3-node graph: Const, Const, Add(c0, c1). Returns the
    /// graph and the Add's id (the realize-root).
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

    /// **Verification 1 — compile_plan walks the graph and skips
    /// kernel-less nodes.** Const inputs + a Reshape (view-shaped op,
    /// op_to_op_kind returns None) get no binding; only the kernel-
    /// bearing nodes (the Add) land in `plan.bindings`.
    #[test]
    fn compile_plan_walks_graph_and_skips_view_and_const_nodes() {
        let mut table = KernelBindingTable::new();
        register(
            &mut table,
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            ok_kernel_a,
            BIT_STABLE,
        );

        // Const, Const, Add, then Reshape on top of the Add — the
        // Reshape is a "no-kernel" node (Op::Reshape isn't in the
        // op_to_op_kind table, so compile_plan skips it).
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
        let plan = compile_plan(&g, &order, &table).expect("compile_plan");

        // 4 nodes in topo order; only the Add gets a binding.
        assert_eq!(plan.order.len(), 4);
        assert_eq!(plan.bindings.len(), 1, "only Add carries a binding");
        let b = plan.binding(add).expect("Add binding present");
        assert_eq!(b.op_kind, OpKind::AddElementwise);
        assert_eq!(
            b.dtypes.as_slice(),
            &[DType::F32, DType::F32, DType::F32],
        );
        assert_eq!(b.backend, BackendId::Cpu);
        assert!(b.kernel.is_none(), "kernel slot stays None at compile_plan time");
        assert!(plan.binding(lhs).is_none(), "Const has no binding");
        assert!(plan.binding(rhs).is_none(), "Const has no binding");
        assert!(
            plan.binding(reshape).is_none(),
            "Reshape (no OpKind mapping) has no binding",
        );
    }

    /// **Verification 2 — compile_plan fails fast on missing binding.**
    /// With no kernel registered for `(MatMul, [F32, F32, F32], Cpu)`,
    /// building a plan for a graph that uses MatMul returns
    /// `Err(NoBackendForOp)` at plan time. The executor never sees
    /// the missing-binding error.
    #[test]
    fn compile_plan_fails_fast_on_missing_binding() {
        // Empty table — no MatMul registration anywhere.
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
        let err = compile_plan(&g, &order, &table).expect_err("plan must error");
        match err {
            fuel_core_types::Error::NoBackendForOp { op, dtypes, .. } => {
                assert_eq!(op, OpKind::MatMul);
                assert_eq!(dtypes, vec![DType::F32, DType::F32, DType::F32]);
            }
            other => panic!("expected NoBackendForOp, got {other:?}"),
        }
    }

    /// **Verification 3 — resolve_kernel caches the first resolution
    /// (lazy).** Second call returns the same KernelRef without
    /// touching the table. We assert idempotency by checking the
    /// returned pointer matches across calls and that the binding's
    /// kernel slot is populated after the first call.
    #[test]
    fn resolve_kernel_lazy_caches_first_resolution() {
        let mut table = KernelBindingTable::new();
        register(
            &mut table,
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            ok_kernel_a,
            BIT_STABLE,
        );

        let mut binding = empty_binding(
            NodeId(42),
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            DeviceLocation::Cpu,
        );
        assert!(binding.kernel.is_none(), "binding starts unresolved");

        let k1 =
            resolve_kernel(&mut binding, &table, TolerancePolicy::BitStableFirst)
                .expect("first resolve");
        assert!(binding.kernel.is_some(), "binding cached after first call");

        let k2 =
            resolve_kernel(&mut binding, &table, TolerancePolicy::BitStableFirst)
                .expect("second resolve");
        assert_eq!(
            k1 as *const () as usize,
            k2 as *const () as usize,
            "second call returns the cached KernelRef",
        );
        assert_eq!(
            binding.kernel.unwrap() as *const () as usize,
            ok_kernel_a as *const () as usize,
            "the cached kernel is the registered one",
        );
    }

    /// **Verification 3a — BitStableFirst picks the bit-stable
    /// alternative when one exists.** Register two alternatives at the
    /// same decision point: the *first* registered is non-bit-stable,
    /// the *second* is bit-stable. BitStableFirst must return the
    /// second.
    #[test]
    fn resolve_kernel_bitstable_first_picks_bitstable_alternative_when_present() {
        let mut table = KernelBindingTable::new();
        register(
            &mut table,
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            ok_kernel_a,
            PrecisionGuarantee::UNAUDITED, // non-bit-stable
        );
        register(
            &mut table,
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            ok_kernel_b,
            BIT_STABLE,
        );

        let mut binding = empty_binding(
            NodeId(0),
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            DeviceLocation::Cpu,
        );
        let chosen = resolve_kernel(&mut binding, &table, TolerancePolicy::BitStableFirst)
            .expect("resolve");
        assert_eq!(
            chosen as *const () as usize,
            ok_kernel_b as *const () as usize,
            "BitStableFirst picks ok_kernel_b (bit-stable), not ok_kernel_a (UNKNOWN)",
        );
    }

    /// **Verification 4 — BitStableFirst falls back to first-registered
    /// when no alternative is bit-stable.** Register two non-bit-stable
    /// alternatives; BitStableFirst returns the first (registration
    /// order).
    #[test]
    fn resolve_kernel_bitstable_first_falls_back_to_first_when_none_bitstable() {
        let mut table = KernelBindingTable::new();
        register(
            &mut table,
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            ok_kernel_a,
            PrecisionGuarantee::UNAUDITED,
        );
        register(
            &mut table,
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            ok_kernel_b,
            PrecisionGuarantee::UNAUDITED,
        );

        let mut binding = empty_binding(
            NodeId(0),
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            DeviceLocation::Cpu,
        );
        let chosen = resolve_kernel(&mut binding, &table, TolerancePolicy::BitStableFirst)
            .expect("resolve");
        assert_eq!(
            chosen as *const () as usize,
            ok_kernel_a as *const () as usize,
            "BitStableFirst falls back to the first-registered alternative",
        );
    }

    /// **Verification 5 — FirstAlternative returns first-registered
    /// regardless of precision.** Same setup as the bit-stable test
    /// (first non-bit-stable, second bit-stable); FirstAlternative
    /// must return the *first* (unlike BitStableFirst which picks the
    /// bit-stable second).
    #[test]
    fn resolve_kernel_first_alternative_policy_returns_first() {
        let mut table = KernelBindingTable::new();
        register(
            &mut table,
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            ok_kernel_a,
            PrecisionGuarantee::UNAUDITED,
        );
        register(
            &mut table,
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            ok_kernel_b,
            BIT_STABLE,
        );

        let mut binding = empty_binding(
            NodeId(0),
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            DeviceLocation::Cpu,
        );
        let chosen =
            resolve_kernel(&mut binding, &table, TolerancePolicy::FirstAlternative)
                .expect("resolve");
        assert_eq!(
            chosen as *const () as usize,
            ok_kernel_a as *const () as usize,
            "FirstAlternative returns first-registered regardless of precision",
        );
    }

    /// **Bonus integration — compile_plan + resolve_kernel end-to-end.**
    /// Build a small Add graph, register the kernel, build a plan, and
    /// resolve the binding. The end-to-end path produces the kernel the
    /// route picker chose, with the binding's slot populated for cache
    /// hits on subsequent dispatches of the same node within this realize.
    #[test]
    fn compile_plan_then_resolve_kernel_end_to_end() {
        let mut table = KernelBindingTable::new();
        register(
            &mut table,
            OpKind::AddElementwise,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
            ok_kernel_a,
            BIT_STABLE,
        );

        let (g, add_id) = build_add_graph();
        let order = topo_order(&g, add_id);
        let mut plan = compile_plan(&g, &order, &table).expect("compile_plan");

        let binding = plan.binding_mut(add_id).expect("Add binding present");
        let kernel = resolve_kernel(binding, &table, TolerancePolicy::default())
            .expect("resolve");
        assert_eq!(
            kernel as *const () as usize,
            ok_kernel_a as *const () as usize,
        );
        assert!(plan.binding(add_id).unwrap().kernel.is_some());
    }
}

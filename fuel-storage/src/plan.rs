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
use fuel_core_types::{DType, DeviceLocation};
use fuel_graph::NodeId;
use smallvec::SmallVec;

use crate::fused::KernelRevisionHash;
use crate::kernel::{KernelDTypes, KernelRef};

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
        // NodeId(0) is fine for type-shape exercise; no graph
        // interaction. Real plan construction flows through
        // compile_plan (A2).
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
}

//! `Candidate` — one (kernel, placement, precision, cost) bundle that
//! the optimizer ranker is considering at a graph decision point.
//!
//! Phase 1.1 of the picker-work arc. A candidate is what the
//! ranker filters and ranks. Today's [`BindingEntry`] is the
//! per-(op, dtypes, backend)-key cousin; a `Candidate` is the
//! per-decision-point view that adds placement (`BackendId` ×
//! `DeviceLocation`), the `OpParams` the kernel will see at runtime,
//! and (in later phases) coupling info that says "this candidate's
//! cost depends on what alternative wins at adjacent decision points."
//!
//! Phase 1.1 ships the shape with `coupling` as an empty `Vec`.
//! Phase 2 populates it when transfer-op insertion lands.

use fuel_core_types::probe::BackendId;
use fuel_core_types::DeviceLocation;

use crate::fused::{CostEstimate, PrecisionGuarantee};
use crate::kernel::{KernelCaps, KernelRef, OpParams};

/// One concrete dispatch alternative the ranker is considering for a
/// graph decision point. Pre-resolved at plan time — by the time a
/// `Candidate` exists, the binding-table lookup has happened and the
/// `KernelRef` is in hand.
///
/// All fields are explicit on purpose: every successor phase (cost
/// composition, executor consumption, runtime selection) reads from
/// these fields by name, and adding a field is preferable to adding
/// a derived getter.
///
/// # Where it comes from
///
/// Phase 1.2 (`compile_plan` successor) builds candidates by walking
/// [`crate::SystemTopology`]-equivalent metadata: for each
/// `DeviceLocation` the graph node may run on, enumerate co-located
/// `BackendId`s, then for each `(op_kind, dtypes, backend)` triple
/// pull every entry from [`crate::kernel::KernelBindingTable::lookup_alternatives`].
/// Each entry contributes one `Candidate`.
#[derive(Clone, Debug)]
pub struct Candidate {
    /// Pre-resolved kernel function pointer.
    pub kernel: KernelRef,
    /// Capability flags the executor reads to decide auto-Contiguize.
    pub caps: KernelCaps,
    /// Which backend this candidate runs on.
    pub backend: BackendId,
    /// Specific device within `backend`.
    pub device: DeviceLocation,
    /// What this kernel claims about its numerical behavior. Hard
    /// filters consume this to enforce user precision floors and
    /// tolerance budgets.
    pub precision: PrecisionGuarantee,
    /// Layer-1 static cost estimate (FLOPs + bandwidth + launch
    /// overhead) computed against the decision point's input shapes.
    /// Phase 1.4's cost composition reads this; Phase 3's Judge
    /// integration refines it with Layer-2 measurements.
    pub static_cost: CostEstimate,
    /// Planner Stage-2 inbound-transfer term: estimated nanoseconds
    /// to move this decision point's inputs from their committed
    /// residencies onto `device`, summed over every input whose
    /// residency differs from `device`. Populated by
    /// [`super::cost::apply_inbound_transfer_costs`] when a
    /// `TransferEstimator` is threaded through `PlanOptions`; zero
    /// when all inputs are co-resident, residency is unknown, or no
    /// estimator is configured. Kept separate from `static_cost` so
    /// Layer-2 Judge refinement (which REPLACES the kernel-time
    /// estimate) never clobbers the transfer term — ranking adds the
    /// two (`composite_ns(static_cost) + inbound_transfer_ns`).
    pub inbound_transfer_ns: u64,
    /// The op-specific parameter payload that will accompany this
    /// kernel call. Cloned here so the candidate is self-contained
    /// for plan-time reasoning. Most ops carry `OpParams::None`.
    pub op_params: OpParams,
    /// Cost adjustments contingent on adjacent decision points'
    /// resolutions — e.g. "this CUDA candidate's cost is +2ms if the
    /// downstream join falls back to CPU." Empty in Phase 1.1; Phase
    /// 2 populates it when transfer-op cost coupling lands. See
    /// architecture v1.0 §04 "Coupling between decisions" for the
    /// long-term shape.
    pub coupling: Vec<CouplingAdjustment>,
    /// Diagnostic tag identifying the kernel's implementation source
    /// (e.g. `"portable-cpu"`, `"aocl"`, `"mkl"`, `"cublas"`).
    /// Copied from the `BindingEntry::kernel_source` of the binding
    /// this candidate was enumerated from. Never used for dispatch —
    /// the picker distinguishes candidates by `kernel` (function-
    /// pointer identity). Surfaces in Judge telemetry and selector
    /// diagnostics so logs can name which kernel won at a decision
    /// point even when several share `(backend, device)`.
    pub kernel_source: &'static str,
}

/// One conditional cost contribution attached to a [`Candidate`].
/// Phase 1.1 ships the struct shape with an inert default; Phase 2's
/// transfer-op insertion is the first session that constructs real
/// instances.
///
/// The conditional is intentionally minimal — `condition` describes
/// the predicate ("downstream decision point N picks a non-co-located
/// backend") and `delta_cost` is the additive correction. Phase 2
/// will likely grow richer predicates (e.g. specific adjacent-node
/// alternative identity); the placeholder is good enough to let the
/// surrounding types stabilize.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CouplingAdjustment {
    /// Opaque tag describing what conditions this adjustment fires
    /// under. Phase 2 will replace this with a richer predicate type;
    /// the current `String` is a stand-in to keep Phase 1.1 from
    /// over-committing on a representation that Phase 2's needs will
    /// shape.
    pub condition: String,
    /// Additive nanosecond correction to the candidate's static cost
    /// when `condition` holds.
    pub delta_ns: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core_types::Layout;
    use fuel_core_types::Result;
    use std::sync::{Arc, RwLock};

    use fuel_storage::Storage;

    fn noop_kernel(
        _inputs: &[Arc<RwLock<Storage>>],
        _outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        _params: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    /// Smoke: candidate construction populates every required field.
    #[test]
    fn candidate_construction() {
        let c = Candidate {
            kernel: noop_kernel,
            caps: KernelCaps::empty(),
            backend: BackendId::Cpu,
            device: DeviceLocation::Cpu,
            precision: PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            static_cost: CostEstimate {
                flops: 100,
                bytes_moved: 400,
                kernel_overhead_ns: 50,
            },
            inbound_transfer_ns: 0,
            op_params: OpParams::None,
            coupling: Vec::new(),
            kernel_source: "",
        };
        assert_eq!(c.backend, BackendId::Cpu);
        assert_eq!(c.static_cost.flops, 100);
        assert!(c.coupling.is_empty());
    }

    /// Smoke: `CouplingAdjustment::default` is inert (empty condition,
    /// zero delta). Phase 2 populates real instances.
    #[test]
    fn coupling_adjustment_default_is_inert() {
        let a = CouplingAdjustment::default();
        assert!(a.condition.is_empty());
        assert_eq!(a.delta_ns, 0);
    }
}

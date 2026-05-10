//! FusedOpRegistry — kernel-side payload. Phase 7.6 step 1 (skeleton).
//!
//! Architecture v1.0 splits the fused-op registry across two crates:
//! - graph-side metadata in `fuel-graph::registry` (id, name, family,
//!   pattern, decompose, backward, shape/dtype rules);
//! - kernel-side payload here in `fuel-storage::fused` ([`BackendImpl`],
//!   [`CostEstimate`], [`PrecisionGuarantee`], [`KernelRevisionHash`]).
//!
//! The split exists because [`KernelRef`] lives in fuel-storage and
//! fuel-graph cannot depend on fuel-storage (the dependency arrow goes
//! the other way). Joining the two halves is by [`fuel_graph::registry::FusedOpId`]
//! at runtime: the optimizer reads the metadata-side entry to reason
//! about decomposition and shape, then asks the kernel-side
//! [`FusedKernelRegistry`] for the per-backend [`BackendImpl`] when it
//! needs to pre-resolve a `KernelRef`.
//!
//! ## Status (step 1)
//!
//! Types only. No callers; no behavior change. Subsequent steps:
//! - Step 3: register the SoftmaxLastDim CPU `BackendImpl` and dispatch
//!   `Op::Fused(SOFTMAX_LAST_DIM, _)` through it from the executor.
//! - Step 6-9: extend per-backend coverage, populate `PrecisionGuarantee`
//!   and `cost`, and migrate the binding-table lookup off the executor's
//!   hot path.

use crate::kernel::{KernelCaps, KernelRef};
use fuel_core_types::{Shape, backend::BackendCapabilities, probe::BackendId};
use fuel_graph::registry::{FusedOpId, FusedOpParams};
use smallvec::SmallVec;
use std::collections::HashMap;

/// Per-backend kernel implementation for one fused op. The optimizer
/// reads this to (1) pre-resolve [`KernelRef`] for nodes it places on
/// this backend, (2) score routes against [`CostEstimate`], and
/// (3) admit candidates against the per-route tolerance budget via
/// [`PrecisionGuarantee`].
///
/// Function-pointer composition (no trait-object indirection) — the
/// registry stores [`BackendImpl`] values inline, the executor calls
/// the function pointer directly.
#[derive(Copy, Clone)]
pub struct BackendImpl {
    /// Dispatch wrapper for this backend's kernel for this fused op.
    /// Same `KernelRef` signature as primitive-op kernels.
    pub kernel: KernelRef,
    /// Cost-estimate function. Given the input shapes, the
    /// per-instance fused-op params, and the backend's capabilities,
    /// returns a [`CostEstimate`] used for placement and route ranking.
    pub cost: fn(&[Shape], &FusedOpParams, &BackendCapabilities) -> CostEstimate,
    /// Numerical precision properties of this kernel.
    pub precision: PrecisionGuarantee,
    /// Layout / capability flags (e.g. strided_input).
    pub caps: KernelCaps,
    /// Revision hash — opaque identifier for the source-version of this
    /// kernel. Persisted optimization caches use this to detect kernel
    /// drift and invalidate stale cache entries (see
    /// `docs/architecture/11-persistence.md`).
    pub revision: KernelRevisionHash,
}

/// Coarse cost model for one kernel invocation. Layer-1 of the
/// architecture's cost-model tower (FLOP counts + bandwidth + launch
/// overhead). Layer-2 — empirical refinement from per-deployment
/// telemetry — composes on top of these without changing this shape.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CostEstimate {
    /// Compute pressure — number of floating-point operations.
    pub flops: u64,
    /// Bandwidth pressure — bytes moved through device memory hierarchy
    /// for this kernel's inputs + outputs (excluding cache hits).
    pub bytes_moved: u64,
    /// Fixed launch overhead. CPU kernels measure this in tens of ns;
    /// GPU launches in low microseconds.
    pub kernel_overhead_ns: u32,
}

/// What this kernel guarantees about its numerical behavior.
///
/// Replaces the binary-flag OracleGrade concept that pre-architecture
/// drafts used; per architecture v1.0 every kernel registration carries
/// a structured precision statement so the optimizer can reason about
/// tolerance budgets and pick comparators for calibration.
///
/// `bit_stable_on_same_hardware` is the strongest property. The
/// always-built backend (fuel-cpu-backend by convention) commits to
/// providing at least one `bit_stable_on_same_hardware: true` kernel
/// per primitive op as the architecture v1.0 coverage commitment;
/// step 7 enforces this via a CI lint.
///
/// The Optional fields encode the bound's flavor (ULP / relative /
/// absolute). Multiple may be present; the optimizer takes the
/// intersection (a budget is admissible only if every populated bound
/// satisfies it). Absent bounds mean "this kernel makes no claim
/// about that flavor."
#[derive(Copy, Clone, Debug)]
pub struct PrecisionGuarantee {
    /// True iff this kernel produces bit-identical output for
    /// bit-identical inputs on the same hardware (no nondeterminism
    /// from kernel scheduling, atomic ordering, etc.).
    pub bit_stable_on_same_hardware: bool,
    /// Maximum unit-in-last-place error vs the IEEE-754 correctly-
    /// rounded result. Tighter than max_relative for low-magnitude
    /// values; many vendor math libraries quote ULPs.
    pub max_ulp: Option<u32>,
    /// Maximum relative error: `|out - ref| / max(|ref|, eps)`.
    pub max_relative: Option<f64>,
    /// Maximum absolute error: `|out - ref|`.
    pub max_absolute: Option<f64>,
    /// Free-text qualifier — implementation hints, vendor citation,
    /// known caveats. Surfaces in error messages; not load-bearing.
    pub notes: &'static str,
}

impl PrecisionGuarantee {
    /// "Reference" guarantee: bit-stable on same hardware, no error
    /// claim above that. Used by reference-grade CPU kernels that
    /// commit to deterministic IEEE-754 evaluation.
    pub const REFERENCE: PrecisionGuarantee = PrecisionGuarantee {
        bit_stable_on_same_hardware: true,
        max_ulp: Some(0),
        max_relative: Some(0.0),
        max_absolute: Some(0.0),
        notes: "Reference IEEE-754 evaluation; bit-identical re-run.",
    };

    /// Conservative all-unknown defaults. Used as a placeholder during
    /// the migration when a real PrecisionGuarantee hasn't been audited
    /// yet. Step 7 replaces every use of this with a real claim and
    /// adds a CI lint that fails when a registered kernel still uses
    /// `UNKNOWN`.
    pub const UNKNOWN: PrecisionGuarantee = PrecisionGuarantee {
        bit_stable_on_same_hardware: false,
        max_ulp: None,
        max_relative: None,
        max_absolute: None,
        notes: "PrecisionGuarantee::UNKNOWN — populate via step 7.",
    };
}

/// Opaque revision hash of a registered kernel. Persisted optimization
/// caches read this to detect kernel drift between cache build and
/// cache load (see `docs/architecture/11-persistence.md`). Computed
/// from kernel source + version metadata at registration time; step 9
/// fills in the actual hashing function alongside the binding-table
/// planning-time refactor.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelRevisionHash(pub u64);

impl KernelRevisionHash {
    /// Sentinel meaning "no revision tracked yet." Used by step-1-shipped
    /// `BackendImpl` registrations until step 9 wires real hashing.
    pub const UNTRACKED: KernelRevisionHash = KernelRevisionHash(0);
}

/// Inline capacity for per-fused-op backend lists. SmallVec at 4 fits
/// CPU + CUDA + Vulkan + Metal without spilling to heap; the typical
/// fused op has 1-3 backends with kernels.
type BackendImplList = SmallVec<[(BackendId, BackendImpl); 4]>;

/// Kernel-side registry: `FusedOpId` → list of per-backend
/// [`BackendImpl`]s. Joined to `fuel-graph`'s metadata-side
/// [`fuel_graph::registry::FusedOpRegistry`] by id at runtime.
///
/// Built at process startup, frozen thereafter (architecture v1.0:
/// no runtime extensibility).
#[derive(Default)]
pub struct FusedKernelRegistry {
    by_id: HashMap<FusedOpId, BackendImplList>,
}

impl FusedKernelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a [`BackendImpl`] for this `(id, backend)` pair. Last
    /// writer wins on duplicate keys (matches today's
    /// `KernelBindingTable::register` idempotency).
    pub fn register(&mut self, id: FusedOpId, backend: BackendId, impl_: BackendImpl) {
        let entry = self.by_id.entry(id).or_default();
        // Replace existing entry for this backend if present.
        if let Some(slot) = entry.iter_mut().find(|(b, _)| *b == backend) {
            slot.1 = impl_;
        } else {
            entry.push((backend, impl_));
        }
    }

    /// Look up the [`BackendImpl`] registered for `(id, backend)`.
    /// Returns `None` when no kernel exists for this combination — the
    /// optimizer's fallback in that case is to lower the fused op via
    /// the metadata-side `decompose` and run it as primitives on the
    /// backend.
    pub fn lookup(&self, id: FusedOpId, backend: BackendId) -> Option<BackendImpl> {
        self.by_id
            .get(&id)
            .and_then(|impls| impls.iter().find(|(b, _)| *b == backend))
            .map(|(_, impl_)| *impl_)
    }

    /// All `(BackendId, BackendImpl)` pairs registered for this id.
    /// The optimizer reads this when deciding placement: a fused op
    /// with a CUDA-only kernel limits its placement candidates.
    pub fn impls_for(&self, id: FusedOpId) -> &[(BackendId, BackendImpl)] {
        self.by_id.get(&id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Number of distinct `FusedOpId`s with at least one registered
    /// backend impl.
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// Whether the registry has any registered impls.
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core_types::{DType, Layout, Result};
    use std::sync::Arc;
    use std::sync::RwLock;

    fn dummy_kernel(
        _inputs: &[Arc<RwLock<crate::Storage>>],
        _outputs: &mut [Arc<RwLock<crate::Storage>>],
        _layouts: &[Layout],
        _params: &crate::kernel::OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn dummy_cost(_s: &[Shape], _p: &FusedOpParams, _c: &BackendCapabilities) -> CostEstimate {
        CostEstimate::default()
    }

    fn make_impl() -> BackendImpl {
        BackendImpl {
            kernel: dummy_kernel,
            cost: dummy_cost,
            precision: PrecisionGuarantee::UNKNOWN,
            caps: KernelCaps::empty(),
            revision: KernelRevisionHash::UNTRACKED,
        }
    }

    /// Smoke: empty registry has no impls.
    #[test]
    fn fused_kernel_registry_empty() {
        let r = FusedKernelRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert!(r.lookup(FusedOpId(1), BackendId::Cpu).is_none());
        assert!(r.impls_for(FusedOpId(1)).is_empty());
        // Suppress unused-warning for DType import on no-feature builds.
        let _ = DType::F32;
    }

    /// Smoke: register and look up a single impl.
    #[test]
    fn fused_kernel_registry_register_and_lookup() {
        let mut r = FusedKernelRegistry::new();
        r.register(FusedOpId(1), BackendId::Cpu, make_impl());
        assert!(!r.is_empty());
        assert_eq!(r.len(), 1);
        let got = r.lookup(FusedOpId(1), BackendId::Cpu);
        assert!(got.is_some());
        let impls = r.impls_for(FusedOpId(1));
        assert_eq!(impls.len(), 1);
        assert_eq!(impls[0].0, BackendId::Cpu);
    }

    /// Re-registering the same (id, backend) overwrites — matches the
    /// existing `KernelBindingTable::register` idempotency.
    #[test]
    fn fused_kernel_registry_register_is_idempotent() {
        let mut r = FusedKernelRegistry::new();
        r.register(FusedOpId(1), BackendId::Cpu, make_impl());
        r.register(FusedOpId(1), BackendId::Cpu, make_impl());
        assert_eq!(r.len(), 1);
        assert_eq!(r.impls_for(FusedOpId(1)).len(), 1);
    }

    /// PrecisionGuarantee::REFERENCE has the strongest properties.
    #[test]
    fn precision_guarantee_reference_is_strict() {
        let p = PrecisionGuarantee::REFERENCE;
        assert!(p.bit_stable_on_same_hardware);
        assert_eq!(p.max_ulp, Some(0));
    }

    /// CostEstimate::default is all-zero.
    #[test]
    fn cost_estimate_default_zero() {
        let c = CostEstimate::default();
        assert_eq!(c.flops, 0);
        assert_eq!(c.bytes_moved, 0);
        assert_eq!(c.kernel_overhead_ns, 0);
    }
}

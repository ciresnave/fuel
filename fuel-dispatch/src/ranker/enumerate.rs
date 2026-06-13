//! Candidate enumeration — walk the binding table for each
//! `(BackendId, DeviceLocation)` placement the caller hands in and
//! produce an [`AlternativeSet`] populated with one [`Candidate`]
//! per registered alternative.
//!
//! Phase 1.2 of the picker-work arc.
//!
//! # Where placements come from
//!
//! This enumerator deliberately does NOT take a SystemTopology
//! reference. The caller (Phase 1.5's `compile_plan` rewrite) is
//! responsible for assembling the placement list — typically via
//! `SystemTopology::backends_for(device)` for cross-co-located-backend
//! picking, or a single `(BackendId, DeviceLocation)` pair when the
//! user has explicitly pinned the node.
//!
//! The slice shape keeps `fuel-dispatch` ignorant of the
//! `SystemTopology` API (which lives in `fuel-core`), avoiding a dep
//! cycle. SystemTopology integration happens at the caller layer.
//!
//! # What gets enumerated
//!
//! For each `(backend, device)` placement, the enumerator queries
//! [`KernelBindingTable::lookup_alternatives`] for every kernel
//! registered against `(op_kind, dtypes, backend)`. Each binding
//! entry contributes one `Candidate`. Backends with no registered
//! kernel for the key contribute zero (silently skipped — that's
//! the binding table saying "this backend doesn't support this op
//! on these dtypes").
//!
//! The resulting set may be:
//!
//! - **Empty** — no candidate placement has a registered kernel.
//!   Callers should treat this as `Error::NoBackendForOp` (the
//!   enumerator doesn't surface the error itself because future
//!   callers may want to fall back to other strategies before
//!   declaring defeat).
//! - **Single-candidate** — one backend, one kernel. The filter
//!   chain becomes trivial.
//! - **Multi-candidate** — the picker has real choices. This is
//!   the working set the rest of Phase 1's filter + rank machinery
//!   operates against.

use fuel_core_types::dispatch::OpKind;
use fuel_core_types::probe::BackendId;
use fuel_core_types::{DType, DeviceLocation};

use super::alternative_set::{AlternativeSet, DEFAULT_MAX_N};
use super::candidate::Candidate;
use crate::kernel::{KernelBindingTable, OpParams};

/// Enumerate candidates for a graph decision point against the
/// supplied placements. See module docs.
///
/// `placements` is the union of `(BackendId, DeviceLocation)` pairs
/// the picker is allowed to consider. Typical sources:
///
/// - User-pinned: a single placement reflecting the user's explicit
///   choice (`Tensor::realize_f32_cuda(&dev)` shape).
/// - SystemTopology-driven: every backend co-located at the node's
///   target `DeviceLocation` (the cross-backend-at-same-device case
///   that unlocks the AOCL/MKL/CPU competition story).
/// - Cross-device: multiple `DeviceLocation`s when the planner
///   wants to compare placements (Phase 2 territory; not in 1.2's
///   scope but the slice shape accommodates it).
///
/// `op_params` is cloned into each `Candidate`. Most ops use
/// `OpParams::None`; cloning is cheap for that variant. Conv2D and
/// similar param-heavy variants pay one clone per candidate, which
/// at plan time is well below the noise floor.
///
/// `max_n` controls the resulting set's truncation bound. Use
/// [`DEFAULT_MAX_N`] (3) unless a caller has a specific reason.
pub fn enumerate_candidates(
    op_kind: OpKind,
    dtypes: &[DType],
    placements: &[(BackendId, DeviceLocation)],
    op_params: &OpParams,
    bindings: &KernelBindingTable,
    max_n: usize,
) -> AlternativeSet {
    let mut set = AlternativeSet::with_max_n(max_n);
    for &(backend, device) in placements {
        for entry in bindings.lookup_alternatives(op_kind, dtypes, backend) {
            set.push(Candidate {
                kernel: entry.kernel,
                caps: entry.caps,
                backend,
                device,
                precision: entry.precision,
                // Phase 1.4 fills in a real cost via entry.cost(...);
                // 1.2 ships the enumerator with placeholder default
                // costs. The cost composer reads back through the
                // candidate's backend/device + binding-table entry
                // when it needs the live CostFn output.
                static_cost: Default::default(),
                // Stage-2 transfer pricing populates this after
                // enumeration (`apply_inbound_transfer_costs`).
                inbound_transfer_ns: 0,
                op_params: op_params.clone(),
                coupling: Vec::new(),
                kernel_source: entry.kernel_source,
            });
        }
    }
    set
}

/// Convenience wrapper around [`enumerate_candidates`] with
/// [`DEFAULT_MAX_N`].
pub fn enumerate_candidates_default(
    op_kind: OpKind,
    dtypes: &[DType],
    placements: &[(BackendId, DeviceLocation)],
    op_params: &OpParams,
    bindings: &KernelBindingTable,
) -> AlternativeSet {
    enumerate_candidates(op_kind, dtypes, placements, op_params, bindings, DEFAULT_MAX_N)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::PrecisionGuarantee;
    use crate::kernel::{unknown_cost, KernelCaps};
    use fuel_core_types::{Layout, Result};
    use fuel_memory::Storage;
    use std::sync::{Arc, RwLock};

    fn noop_a(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn noop_b(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn noop_c(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn one_dtype_key() -> [DType; 3] {
        [DType::F32, DType::F32, DType::F32]
    }

    fn empty_table() -> KernelBindingTable {
        KernelBindingTable::new()
    }

    fn table_with(entries: &[(BackendId, fn(&[Arc<RwLock<Storage>>], &mut [Arc<RwLock<Storage>>], &[Layout], &OpParams) -> Result<()>)]) -> KernelBindingTable {
        let mut t = KernelBindingTable::new();
        for &(backend, kernel) in entries {
            t.register_full(
                OpKind::AddElementwise,
                &one_dtype_key(),
                backend,
                kernel,
                KernelCaps::empty(),
                PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
                unknown_cost,
            );
        }
        t
    }

    #[test]
    fn empty_placements_yields_empty_set() {
        let bindings = table_with(&[(BackendId::Cpu, noop_a)]);
        let set = enumerate_candidates_default(
            OpKind::AddElementwise,
            &one_dtype_key(),
            &[],
            &OpParams::None,
            &bindings,
        );
        assert!(set.is_empty());
    }

    #[test]
    fn empty_binding_table_yields_empty_set() {
        let bindings = empty_table();
        let set = enumerate_candidates_default(
            OpKind::AddElementwise,
            &one_dtype_key(),
            &[(BackendId::Cpu, DeviceLocation::Cpu)],
            &OpParams::None,
            &bindings,
        );
        assert!(set.is_empty());
    }

    #[test]
    fn single_placement_single_alternative() {
        let bindings = table_with(&[(BackendId::Cpu, noop_a)]);
        let set = enumerate_candidates_default(
            OpKind::AddElementwise,
            &one_dtype_key(),
            &[(BackendId::Cpu, DeviceLocation::Cpu)],
            &OpParams::None,
            &bindings,
        );
        assert_eq!(set.len(), 1);
        let c = set.winner().unwrap();
        assert_eq!(c.backend, BackendId::Cpu);
        assert_eq!(c.device, DeviceLocation::Cpu);
    }

    #[test]
    fn cross_co_located_backends_aggregated() {
        // Aocl + Mkl + Cpu all at DeviceLocation::Cpu. Each
        // contributes one alternative — the classic vendor-CPU
        // multi-backend story.
        let bindings = table_with(&[
            (BackendId::Cpu, noop_a),
            (BackendId::Cuda, noop_b),
            (BackendId::Vulkan, noop_c),
        ]);
        let set = enumerate_candidates_default(
            OpKind::AddElementwise,
            &one_dtype_key(),
            &[
                (BackendId::Cpu, DeviceLocation::Cpu),
                (BackendId::Cuda, DeviceLocation::Cpu),
                (BackendId::Vulkan, DeviceLocation::Cpu),
            ],
            &OpParams::None,
            &bindings,
        );
        assert_eq!(set.len(), 3);
        let backends: Vec<BackendId> = set
            .alternatives()
            .iter()
            .map(|c| c.backend)
            .collect();
        assert_eq!(backends, vec![BackendId::Cpu, BackendId::Cuda, BackendId::Vulkan]);
        // All three are at DeviceLocation::Cpu — the CPU storage
        // substrate is shared so no Op::Copy is needed between them.
        assert!(set.alternatives().iter().all(|c| c.device == DeviceLocation::Cpu));
    }

    #[test]
    fn multi_alt_per_backend_within_one_placement() {
        // One backend, two kernel registrations at the same
        // decision-point key. The cuBLAS+CUTLASS-as-sibling pattern
        // architecture v1.0 §04 names.
        let mut bindings = KernelBindingTable::new();
        bindings.register_full(
            OpKind::AddElementwise,
            &one_dtype_key(),
            BackendId::Cpu,
            noop_a,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            unknown_cost,
        );
        bindings.register_full(
            OpKind::AddElementwise,
            &one_dtype_key(),
            BackendId::Cpu,
            noop_b,
            KernelCaps::empty(),
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            unknown_cost,
        );
        let set = enumerate_candidates_default(
            OpKind::AddElementwise,
            &one_dtype_key(),
            &[(BackendId::Cpu, DeviceLocation::Cpu)],
            &OpParams::None,
            &bindings,
        );
        assert_eq!(set.len(), 2, "both alternatives at the same key surface");
    }

    #[test]
    fn placement_with_no_kernel_silently_contributes_zero() {
        // CUDA backend present in placement list but no kernel
        // registered. CPU has one. Result is a 1-candidate set with
        // the CPU one only.
        let bindings = table_with(&[(BackendId::Cpu, noop_a)]);
        let set = enumerate_candidates_default(
            OpKind::AddElementwise,
            &one_dtype_key(),
            &[
                (BackendId::Cuda, DeviceLocation::Cuda { gpu_id: 0 }),
                (BackendId::Cpu, DeviceLocation::Cpu),
            ],
            &OpParams::None,
            &bindings,
        );
        assert_eq!(set.len(), 1);
        assert_eq!(set.winner().unwrap().backend, BackendId::Cpu);
    }

    #[test]
    fn max_n_propagates_into_set() {
        let bindings = table_with(&[(BackendId::Cpu, noop_a)]);
        let set = enumerate_candidates(
            OpKind::AddElementwise,
            &one_dtype_key(),
            &[(BackendId::Cpu, DeviceLocation::Cpu)],
            &OpParams::None,
            &bindings,
            5,
        );
        assert_eq!(set.max_n(), 5);
    }

    #[test]
    fn op_params_cloned_onto_every_candidate() {
        let bindings = table_with(&[
            (BackendId::Cpu, noop_a),
            (BackendId::Cuda, noop_b),
        ]);
        // Use a non-None variant to verify the clone is actually
        // happening rather than every candidate getting the trivial
        // default.
        let params = OpParams::Reduce { dims: vec![0, 2], keepdim: false };
        let set = enumerate_candidates_default(
            OpKind::AddElementwise,
            &one_dtype_key(),
            &[
                (BackendId::Cpu, DeviceLocation::Cpu),
                (BackendId::Cuda, DeviceLocation::Cpu),
            ],
            &params,
            &bindings,
        );
        assert_eq!(set.len(), 2);
        for c in set.alternatives() {
            match &c.op_params {
                OpParams::Reduce { dims, keepdim } => {
                    assert_eq!(dims, &vec![0, 2]);
                    assert!(!*keepdim);
                }
                other => panic!("expected Reduce, got {other:?}"),
            }
        }
    }
}

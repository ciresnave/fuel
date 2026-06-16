//! Concrete [`AlternativeFilter`] implementations shipped in
//! Phase 1.3 of the picker-work arc.
//!
//! ## Default chain composition
//!
//! Callers that don't have application-specific filter needs can
//! use [`default_chain`] to assemble the recommended chain:
//!
//! 1. [`PrecisionFloorFilter`] (hard) — drops candidates that fail
//!    the user's per-call precision requirement.
//! 2. [`StridedInputPreferenceFilter`] (soft, `min_remaining = 1`)
//!    — prefers `caps.strided_input` when any input is non-contiguous;
//!    no-op otherwise.
//! 3. [`BitStablePreferenceFilter`] (soft, `min_remaining = 1`) —
//!    prefers bit-stable candidates when available without strictly
//!    requiring them.
//!
//! Phase 3 will insert a Judge-driven `JudgeFastestFilter` (soft,
//! `min_remaining = 1`) at the end of this chain when empirical
//! data is available.
//!
//! [`AlternativeFilter`]: super::filter::AlternativeFilter

pub mod bit_stable_pref;
pub mod precision_floor;
pub mod strided_input_pref;

pub use bit_stable_pref::BitStablePreferenceFilter;
pub use precision_floor::{PrecisionFloorFilter, PrecisionRequirement};
pub use strided_input_pref::StridedInputPreferenceFilter;

use super::filter::AlternativeFilter;

/// Build the recommended default filter chain.
///
/// `precision_requirement` defaults to unconstrained
/// ([`PrecisionRequirement::default`]); callers tighten it for
/// determinism-sensitive realizes (training, golden tests).
pub fn default_chain(
    precision_requirement: PrecisionRequirement,
) -> Vec<Box<dyn AlternativeFilter>> {
    vec![
        Box::new(PrecisionFloorFilter::new(precision_requirement)),
        Box::new(StridedInputPreferenceFilter::default()),
        Box::new(BitStablePreferenceFilter::default()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::PrecisionGuarantee;
    use crate::kernel::{KernelCaps, OpParams};
    use crate::ranker::alternative_set::AlternativeSet;
    use crate::ranker::candidate::Candidate;
    use crate::ranker::chain::apply_filter_chain;
    use crate::ranker::filter::FilterContext;
    use fuel_core_types::dispatch::OpKind;
    use fuel_core_types::probe::BackendId;
    use fuel_core_types::{DType, DeviceLocation, Layout, Result, Shape};
    use fuel_memory::Storage;
    use std::sync::{Arc, RwLock};

    fn noop(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn candidate(bit_stable: bool, strided_caps: bool) -> Candidate {
        Candidate {
            kernel: noop,
            caps: if strided_caps {
                KernelCaps::strided_input()
            } else {
                KernelCaps::empty()
            },
            backend: BackendId::Cpu,
            device: DeviceLocation::Cpu,
            precision: PrecisionGuarantee {
                bit_stable_on_same_hardware: bit_stable,
                ..PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU
            },
            static_cost: Default::default(),
            inbound_transfer_ns: 0,
            op_params: OpParams::None,
            coupling: Vec::new(),
            kernel_source: "",
        }
    }

    fn ctx<'a>() -> FilterContext<'a> {
        static LAYOUTS: std::sync::OnceLock<Vec<Layout>> = std::sync::OnceLock::new();
        let layouts = LAYOUTS.get_or_init(|| {
            vec![Layout::contiguous(Shape::from(vec![4]))]
        });
        FilterContext::new(OpKind::AddElementwise, &[DType::F32], layouts)
    }

    #[test]
    fn default_chain_unconstrained_passes_everything() {
        let chain = default_chain(PrecisionRequirement::default());
        let mut set = AlternativeSet::from_candidates(
            vec![
                candidate(false, false),
                candidate(true, false),
                candidate(false, true),
            ],
        );
        apply_filter_chain(&mut set, &chain, &ctx()).expect("chain");
        // Precision floor unconstrained → all pass.
        // Strided-input pref: input is contiguous → no-op.
        // Bit-stable pref (soft, min=1): one candidate is bit-stable,
        // so the filter applies and keeps just it.
        assert_eq!(set.len(), 1);
        assert!(set.winner().unwrap().precision.bit_stable_on_same_hardware);
    }

    #[test]
    fn default_chain_bit_stable_required_drops_unstable() {
        let chain = default_chain(PrecisionRequirement::BIT_STABLE);
        let mut set = AlternativeSet::from_candidates(
            vec![candidate(false, false), candidate(true, false)],
        );
        apply_filter_chain(&mut set, &chain, &ctx()).expect("chain");
        assert_eq!(set.len(), 1);
        assert!(set.winner().unwrap().precision.bit_stable_on_same_hardware);
    }

    #[test]
    fn default_chain_no_bit_stable_falls_back() {
        // No bit-stable candidate exists. With BIT_STABLE required,
        // the hard precision-floor filter empties the set →
        // FilterRejected.
        let chain = default_chain(PrecisionRequirement::BIT_STABLE);
        let mut set = AlternativeSet::from_candidates(
            vec![candidate(false, false), candidate(false, true)],
        );
        let err = apply_filter_chain(&mut set, &chain, &ctx()).unwrap_err();
        match err {
            fuel_core_types::Error::FilterRejected { filter, .. } => {
                assert_eq!(filter, "precision-floor");
            }
            other => panic!("expected FilterRejected, got {other:?}"),
        }
    }

    #[test]
    fn default_chain_unconstrained_no_bit_stable_keeps_all() {
        // No bit-stable, precision unconstrained → bit-stable pref
        // is soft → it returns empty → skipped → all survive.
        let chain = default_chain(PrecisionRequirement::default());
        let mut set = AlternativeSet::from_candidates(
            vec![candidate(false, false), candidate(false, true)],
        );
        apply_filter_chain(&mut set, &chain, &ctx()).expect("chain");
        assert_eq!(set.len(), 2, "no bit-stable + soft pref → no narrowing");
    }
}

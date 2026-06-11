//! Hard filter: drop candidates whose `PrecisionGuarantee` doesn't
//! meet the user's `PrecisionRequirement`.
//!
//! Phase 1.3 of the picker-work arc.
//!
//! # Comparison semantics
//!
//! The requirement is a per-bound floor. For each populated bound:
//!
//! - `require_bit_stable: true` → candidate's
//!   `precision.bit_stable_on_same_hardware` must be `true`.
//! - `max_ulp: Some(n)` → candidate's `precision.max_ulp` must be
//!   `Some(k)` with `k ≤ n`. A candidate with `max_ulp: None` (no
//!   claim) fails the bound because we can't prove admissibility.
//! - Same shape for `max_relative` and `max_absolute`.
//!
//! An empty `PrecisionRequirement` (all fields `None`/`false`) is
//! the no-op default — every candidate passes. The filter is still
//! classified `Hard` so callers who want zero-tolerance-for-misses
//! get the consistent `Error::FilterRejected` shape; soft variants
//! can be added later if needed.
//!
//! # Why "no claim" fails a bound
//!
//! A `PrecisionGuarantee` with `max_ulp: None` means the kernel
//! author hasn't asserted a ULP bound. That's distinct from "the
//! kernel is bad" — it just means we can't *prove* it meets the
//! user's floor. Per architecture v1.0 §07 (tolerance), the
//! conservative answer to "is this kernel admissible?" when the
//! claim is silent is "no." Future audited claims fill in the
//! bounds; until then, kernels that haven't been audited can't
//! pass an explicit floor request.

use crate::fused::PrecisionGuarantee;
use crate::ranker::candidate::Candidate;
use crate::ranker::filter::{AlternativeFilter, FilterClass, FilterContext};

/// User-supplied per-bound floor for the precision filter.
///
/// All fields default to "don't care" (no constraint). Setting one
/// or more imposes a hard requirement — candidates that can't
/// prove they meet it are dropped, and if every candidate fails
/// the filter chain raises `Error::FilterRejected`.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct PrecisionRequirement {
    /// Require the kernel to be bit-stable on same hardware.
    pub require_bit_stable: bool,
    /// Maximum tolerated ULP error vs IEEE-754 correctly-rounded
    /// result.
    pub max_ulp: Option<u32>,
    /// Maximum tolerated relative error.
    pub max_relative: Option<f64>,
    /// Maximum tolerated absolute error.
    pub max_absolute: Option<f64>,
}

impl PrecisionRequirement {
    /// "Bit-stable required" — the strictest common preset. Used by
    /// determinism-critical applications (training reproducibility,
    /// CI golden tests). Equivalent to architecture v1.0 §07's
    /// `Strict` tolerance budget at the call-site granularity.
    pub const BIT_STABLE: PrecisionRequirement = PrecisionRequirement {
        require_bit_stable: true,
        max_ulp: None,
        max_relative: None,
        max_absolute: None,
    };

    /// Is this requirement a no-op (no constraints set)?
    pub fn is_unconstrained(&self) -> bool {
        !self.require_bit_stable
            && self.max_ulp.is_none()
            && self.max_relative.is_none()
            && self.max_absolute.is_none()
    }

    /// Does `precision` satisfy every set bound in this requirement?
    pub fn admits(&self, precision: &PrecisionGuarantee) -> bool {
        if self.require_bit_stable && !precision.bit_stable_on_same_hardware {
            return false;
        }
        if let Some(floor) = self.max_ulp {
            match precision.max_ulp {
                Some(claim) if claim <= floor => {}
                _ => return false,
            }
        }
        if let Some(floor) = self.max_relative {
            match precision.max_relative {
                Some(claim) if claim <= floor => {}
                _ => return false,
            }
        }
        if let Some(floor) = self.max_absolute {
            match precision.max_absolute {
                Some(claim) if claim <= floor => {}
                _ => return false,
            }
        }
        true
    }
}

/// The filter itself. Wraps a `PrecisionRequirement` and surfaces
/// it through the [`AlternativeFilter`] trait.
#[derive(Copy, Clone, Debug)]
pub struct PrecisionFloorFilter {
    pub requirement: PrecisionRequirement,
}

impl PrecisionFloorFilter {
    pub fn new(requirement: PrecisionRequirement) -> Self {
        Self { requirement }
    }
}

impl AlternativeFilter for PrecisionFloorFilter {
    fn filter(&self, alts: &[Candidate], _ctx: &FilterContext) -> Vec<usize> {
        if self.requirement.is_unconstrained() {
            // No-op fast path. Common: most realizes don't set a
            // floor and the chain has this filter as a placeholder.
            return (0..alts.len()).collect();
        }
        alts.iter()
            .enumerate()
            .filter_map(|(i, c)| self.requirement.admits(&c.precision).then_some(i))
            .collect()
    }

    fn classification(&self) -> FilterClass {
        FilterClass::Hard
    }

    fn name(&self) -> &'static str {
        "precision-floor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::PrecisionGuarantee;
    use crate::kernel::{KernelCaps, OpParams};
    use crate::ranker::candidate::Candidate;
    use fuel_core_types::dispatch::OpKind;
    use fuel_core_types::probe::BackendId;
    use fuel_core_types::{DType, DeviceLocation, Layout, Result, Shape};
    use fuel_storage::Storage;
    use std::sync::{Arc, RwLock};

    fn noop(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn candidate_with_precision(precision: PrecisionGuarantee) -> Candidate {
        Candidate {
            kernel: noop,
            caps: KernelCaps::empty(),
            backend: BackendId::Cpu,
            device: DeviceLocation::Cpu,
            precision,
            static_cost: Default::default(),
            inbound_transfer_ns: 0,
            op_params: OpParams::None,
            coupling: Vec::new(),
            kernel_source: "",
        }
    }

    fn ctx<'a>() -> FilterContext<'a> {
        static LAYOUTS: std::sync::OnceLock<Vec<Layout>> = std::sync::OnceLock::new();
        let layouts = LAYOUTS.get_or_init(|| vec![Layout::contiguous(Shape::from(vec![4]))]);
        FilterContext::new(OpKind::AddElementwise, &[DType::F32], layouts)
    }

    #[test]
    fn unconstrained_admits_everything() {
        let req = PrecisionRequirement::default();
        assert!(req.is_unconstrained());
        let f = PrecisionFloorFilter::new(req);
        let alts = vec![
            candidate_with_precision(PrecisionGuarantee::UNAUDITED),
            candidate_with_precision(PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU),
            candidate_with_precision(PrecisionGuarantee::REFERENCE),
        ];
        assert_eq!(f.filter(&alts, &ctx()), vec![0, 1, 2]);
    }

    #[test]
    fn bit_stable_required_drops_non_bit_stable() {
        let f = PrecisionFloorFilter::new(PrecisionRequirement::BIT_STABLE);
        let alts = vec![
            candidate_with_precision(PrecisionGuarantee {
                bit_stable_on_same_hardware: false,
                ..PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU
            }),
            candidate_with_precision(PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU),
        ];
        assert_eq!(f.filter(&alts, &ctx()), vec![1]);
    }

    #[test]
    fn max_ulp_requires_at_or_below() {
        let req = PrecisionRequirement {
            max_ulp: Some(2),
            ..Default::default()
        };
        let f = PrecisionFloorFilter::new(req);
        let alts = vec![
            candidate_with_precision(PrecisionGuarantee {
                max_ulp: Some(0),
                ..PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU
            }),
            candidate_with_precision(PrecisionGuarantee {
                max_ulp: Some(2),
                ..PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU
            }),
            candidate_with_precision(PrecisionGuarantee {
                max_ulp: Some(3),
                ..PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU
            }),
        ];
        assert_eq!(f.filter(&alts, &ctx()), vec![0, 1]);
    }

    #[test]
    fn no_claim_fails_explicit_bound() {
        // max_ulp = Some(5) required, but a candidate with no claim
        // (None) is not admissible — we can't prove it satisfies.
        let req = PrecisionRequirement {
            max_ulp: Some(5),
            ..Default::default()
        };
        let f = PrecisionFloorFilter::new(req);
        let alts = vec![candidate_with_precision(
            PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,  // max_ulp: None
        )];
        assert!(f.filter(&alts, &ctx()).is_empty());
    }

    #[test]
    fn max_relative_floor_enforced() {
        let req = PrecisionRequirement {
            max_relative: Some(1e-3),
            ..Default::default()
        };
        let f = PrecisionFloorFilter::new(req);
        let alts = vec![
            candidate_with_precision(PrecisionGuarantee {
                max_relative: Some(1e-4),
                ..PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU
            }),
            candidate_with_precision(PrecisionGuarantee {
                max_relative: Some(1e-2),
                ..PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU
            }),
        ];
        assert_eq!(f.filter(&alts, &ctx()), vec![0]);
    }

    #[test]
    fn multiple_bounds_intersect() {
        // Both max_ulp and bit_stable required — candidate must
        // satisfy BOTH.
        let req = PrecisionRequirement {
            require_bit_stable: true,
            max_ulp: Some(1),
            ..Default::default()
        };
        let f = PrecisionFloorFilter::new(req);
        let alts = vec![
            // Bit-stable + ulp=0 → passes both.
            candidate_with_precision(PrecisionGuarantee {
                bit_stable_on_same_hardware: true,
                max_ulp: Some(0),
                max_relative: None,
                max_absolute: None,
                notes: "ok",
            }),
            // Bit-stable + ulp=5 → fails ulp.
            candidate_with_precision(PrecisionGuarantee {
                bit_stable_on_same_hardware: true,
                max_ulp: Some(5),
                max_relative: None,
                max_absolute: None,
                notes: "too loose",
            }),
            // Not bit-stable + ulp=0 → fails bit-stable.
            candidate_with_precision(PrecisionGuarantee {
                bit_stable_on_same_hardware: false,
                max_ulp: Some(0),
                max_relative: None,
                max_absolute: None,
                notes: "not stable",
            }),
        ];
        assert_eq!(f.filter(&alts, &ctx()), vec![0]);
    }

    #[test]
    fn filter_class_is_hard() {
        let f = PrecisionFloorFilter::new(PrecisionRequirement::BIT_STABLE);
        assert_eq!(f.classification(), FilterClass::Hard);
        assert_eq!(f.name(), "precision-floor");
    }
}

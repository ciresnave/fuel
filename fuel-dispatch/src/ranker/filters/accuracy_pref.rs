// SPDX-License-Identifier: MIT OR Apache-2.0
//! Soft preference: by default, place by cost among kernels whose accuracy claim meets ONE
//! named bound ([`DEFAULT_ACCURACY_BOUND`]). Board 240, option D (CireSnave, 2026-10-22).
//!
//! # The rule
//!
//! A candidate *meets the default accuracy bound* when either
//!
//! 1. it is **reference-grade**: `bit_stable_on_same_hardware` is true. A bit-stable kernel is
//!    what §07's strict-equivalence default means (the always-built CPU kernels, audited CUDA
//!    kernels); their contracts carry no numeric bound, and treating them as unclaimed would
//!    drop the reference kernels from every comparison; or
//! 2. it **claims** at least one of `max_relative` / `max_absolute`, and every populated claim
//!    of those two flavors is within the bound's value for that flavor. A claim in another
//!    flavor only (`max_ulp`) is not comparable and does not meet the bound.
//!
//! A kernel that is neither (no accuracy claim, not bit-stable) does **not** win by default.
//! The filter is soft with `min_remaining = 1`: when no candidate meets the bound the filter
//! is skipped, so an op whose only kernels are unclaimed still runs (availability beats
//! preference). Cost ranking then picks among the survivors, across devices.
//!
//! # What this is not
//!
//! Repeatability is a separate opt-in: [`super::PrecisionFloorFilter`] with
//! `require_bit_stable: true` (strict reproducibility mode) is unchanged, and still excludes
//! non-bit-stable kernels everywhere. The bound here is an *accuracy* bound; it says nothing
//! about run-to-run bit identity, which a claimed non-bit-stable kernel (the Vulkan matmul)
//! does not have.
//!
//! This replaces `BitStablePreferenceFilter`, which dropped every non-bit-stable candidate
//! across devices before cost ranking and so kept a faster, accurate GPU kernel from ever being
//! compared with the CPU kernel on cost (Qwen3-0.6B Vulkan decode: 253 MatMuls on the CPU,
//! 0.23 tok/s; see `docs/research/qwen3-vulkan-decode-profile.md`).

use crate::fused::PrecisionGuarantee;
use crate::ranker::candidate::Candidate;
use crate::ranker::filter::{AlternativeFilter, FilterClass, FilterContext};

/// The accuracy a kernel's claim must meet to win placement by default. Both components are
/// ceilings on the kernel's claimed error.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AccuracyBound {
    /// Ceiling on a claimed `max_relative` (`|out - ref| / max(|ref|, eps)`).
    pub max_relative: f64,
    /// Ceiling on a claimed `max_absolute` (`|out - ref|`, for unit-scale inputs).
    pub max_absolute: f64,
}

/// THE default accuracy bound. One place, one value.
///
/// `1e-3` relative is the `Tolerance::Relative(1e-3)` ("at most 0.1% deviation per element")
/// example in `docs/architecture/07-tolerance.md`; `1e-3` absolute is the pessimistic bound the
/// CUDA flash-attention contract already carries. Both sit about an order of magnitude above
/// what the Vulkan f32 matmul measures (`fkc::verify::seed_vulkan_matmul_accuracy`, ledger evidence:
/// 2.3e-4 absolute, 8.8e-5 relative over 12 shapes at `k` up to 4096) and about two orders
/// below the 2e-2 logit-level tier the model-level yardstick uses for f16 and quantized
/// weights, so one kernel at the bound cannot by itself breach that tier.
pub const DEFAULT_ACCURACY_BOUND: AccuracyBound = AccuracyBound {
    max_relative: 1.0e-3,
    max_absolute: 1.0e-3,
};

/// Whether `p` meets `bound` under the rule in the module docs.
pub fn meets_accuracy_bound(p: &PrecisionGuarantee, bound: &AccuracyBound) -> bool {
    if p.bit_stable_on_same_hardware {
        return true;
    }
    let rel_ok = p.max_relative.map(|r| r <= bound.max_relative);
    let abs_ok = p.max_absolute.map(|a| a <= bound.max_absolute);
    if rel_ok.is_none() && abs_ok.is_none() {
        return false;
    }
    rel_ok.unwrap_or(true) && abs_ok.unwrap_or(true)
}

/// Soft filter that keeps only candidates meeting [`AccuracyBound`], as long as doing so leaves
/// at least `min_remaining` alternatives in play.
#[derive(Copy, Clone, Debug)]
pub struct AccuracyPreferenceFilter {
    pub min_remaining: usize,
    pub bound: AccuracyBound,
}

impl Default for AccuracyPreferenceFilter {
    fn default() -> Self {
        Self {
            min_remaining: 1,
            bound: DEFAULT_ACCURACY_BOUND,
        }
    }
}

impl AlternativeFilter for AccuracyPreferenceFilter {
    fn filter(&self, alts: &[Candidate], _ctx: &FilterContext) -> Vec<usize> {
        alts.iter()
            .enumerate()
            .filter_map(|(i, c)| meets_accuracy_bound(&c.precision, &self.bound).then_some(i))
            .collect()
    }

    fn classification(&self) -> FilterClass {
        FilterClass::Soft {
            min_remaining: self.min_remaining,
        }
    }

    fn name(&self) -> &'static str {
        "accuracy-preference"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::PrecisionGuarantee;
    use crate::kernel::{KernelCaps, OpParams};
    use crate::ranker::alternative_set::AlternativeSet;
    use crate::ranker::chain::apply_filter_chain;
    use crate::ranker::filters::{PrecisionFloorFilter, PrecisionRequirement, default_chain};
    use fuel_ir::dispatch::OpKind;
    use fuel_ir::probe::BackendId;
    use fuel_ir::{DType, DeviceLocation, Layout, Result, Shape};
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

    fn precision(
        bit_stable: bool,
        rel: Option<f64>,
        abs: Option<f64>,
        ulp: Option<u32>,
    ) -> PrecisionGuarantee {
        PrecisionGuarantee {
            bit_stable_on_same_hardware: bit_stable,
            max_ulp: ulp,
            max_relative: rel,
            max_absolute: abs,
            notes: "test",
        }
    }

    /// `ns` is the whole static cost, so cost ranking order is the `ns` order.
    fn candidate(
        backend: BackendId,
        device: DeviceLocation,
        precision: PrecisionGuarantee,
        ns: u32,
    ) -> Candidate {
        Candidate {
            kernel: noop,
            caps: KernelCaps::empty(),
            backend,
            device,
            precision,
            static_cost: crate::fused::CostEstimate {
                flops: 0,
                bytes_moved: 0,
                kernel_overhead_ns: ns,
            },
            inbound_transfer_ns: 0,
            op_params: OpParams::None,
            coupling: Vec::new(),
            kernel_source: "",
            kernel_revision_hash: 0,
        }
    }

    fn ctx<'a>() -> FilterContext<'a> {
        static LAYOUTS: std::sync::OnceLock<Vec<Layout>> = std::sync::OnceLock::new();
        let layouts = LAYOUTS.get_or_init(|| vec![Layout::contiguous(Shape::from(vec![4]))]);
        FilterContext::new(OpKind::MatMul, &[DType::F32], layouts)
    }

    fn cpu_reference(ns: u32) -> Candidate {
        candidate(
            BackendId::Cpu,
            DeviceLocation::Cpu,
            precision(true, None, None, None),
            ns,
        )
    }

    fn vulkan(p: PrecisionGuarantee, ns: u32) -> Candidate {
        candidate(BackendId::Vulkan, DeviceLocation::Cpu, p, ns)
    }

    /// Run the default chain and cost-rank the survivors; return the winner's backend.
    fn winner_backend(cands: Vec<Candidate>) -> BackendId {
        let mut set = AlternativeSet::from_candidates(cands);
        let chain = default_chain(PrecisionRequirement::default());
        apply_filter_chain(&mut set, &chain, &ctx()).expect("default chain");
        set.rank_by_cost();
        set.winner().expect("a winner").backend
    }

    #[test]
    fn claim_within_bound_wins_on_cost_against_the_reference_kernel() {
        let claimed = precision(false, Some(7.8e-5), Some(2.5e-4), None);
        assert_eq!(
            winner_backend(vec![cpu_reference(1_000), vulkan(claimed, 10)]),
            BackendId::Vulkan,
            "a claimed-accurate GPU kernel that is cheaper wins"
        );
    }

    #[test]
    fn claim_within_bound_still_loses_when_the_reference_kernel_is_cheaper() {
        let claimed = precision(false, Some(7.8e-5), Some(2.5e-4), None);
        assert_eq!(
            winner_backend(vec![cpu_reference(10), vulkan(claimed, 1_000)]),
            BackendId::Cpu,
            "the bound admits a kernel to the cost comparison; it does not hand it the win"
        );
    }

    #[test]
    fn claim_above_bound_loses_even_when_cheaper() {
        let too_loose = precision(false, Some(5.0e-2), Some(2.5e-4), None);
        assert_eq!(
            winner_backend(vec![cpu_reference(1_000), vulkan(too_loose, 10)]),
            BackendId::Cpu,
            "one populated flavor above the bound disqualifies"
        );
        let too_loose_abs = precision(false, Some(7.8e-5), Some(0.5), None);
        assert_eq!(
            winner_backend(vec![cpu_reference(1_000), vulkan(too_loose_abs, 10)]),
            BackendId::Cpu,
        );
    }

    #[test]
    fn no_claim_loses_even_when_cheaper() {
        let unclaimed = precision(false, None, None, None);
        assert_eq!(
            winner_backend(vec![cpu_reference(1_000), vulkan(unclaimed, 10)]),
            BackendId::Cpu,
            "a kernel with no accuracy claim does not win by default"
        );
        let ulp_only = precision(false, None, None, Some(1));
        assert_eq!(
            winner_backend(vec![cpu_reference(1_000), vulkan(ulp_only, 10)]),
            BackendId::Cpu,
            "a ulp-only claim is not comparable to the bound"
        );
    }

    #[test]
    fn a_nan_claim_does_not_meet_the_bound() {
        let nan = precision(false, Some(f64::NAN), None, None);
        assert!(!meets_accuracy_bound(&nan, &DEFAULT_ACCURACY_BOUND));
    }

    #[test]
    fn the_bound_is_inclusive() {
        let at = precision(
            false,
            Some(DEFAULT_ACCURACY_BOUND.max_relative),
            Some(DEFAULT_ACCURACY_BOUND.max_absolute),
            None,
        );
        assert!(meets_accuracy_bound(&at, &DEFAULT_ACCURACY_BOUND));
    }

    #[test]
    fn availability_beats_preference_when_nothing_meets_the_bound() {
        let unclaimed = precision(false, None, None, None);
        let mut set =
            AlternativeSet::from_candidates(vec![vulkan(unclaimed, 10), vulkan(unclaimed, 20)]);
        let chain = default_chain(PrecisionRequirement::default());
        apply_filter_chain(&mut set, &chain, &ctx()).expect("soft skip succeeds");
        assert_eq!(set.len(), 2, "no candidate meets the bound: filter skipped");
    }

    #[test]
    fn strict_mode_is_unchanged_and_still_excludes_non_bit_stable_kernels() {
        // A claimed, cheaper, non-bit-stable GPU kernel: the default chain lets it win, the
        // strict requirement (require_bit_stable) still drops it.
        let claimed = precision(false, Some(7.8e-5), Some(2.5e-4), None);
        let cands = vec![cpu_reference(1_000), vulkan(claimed, 10)];

        let mut strict = AlternativeSet::from_candidates(cands.clone());
        let chain = default_chain(PrecisionRequirement::BIT_STABLE);
        apply_filter_chain(&mut strict, &chain, &ctx()).expect("strict chain");
        strict.rank_by_cost();
        assert_eq!(strict.winner().unwrap().backend, BackendId::Cpu);
        assert_eq!(strict.len(), 1, "the non-bit-stable kernel is excluded");

        // The floor filter itself, on its own, as before.
        let mut only_floor = AlternativeSet::from_candidates(cands);
        let floor: Vec<Box<dyn AlternativeFilter>> = vec![Box::new(PrecisionFloorFilter::new(
            PrecisionRequirement::BIT_STABLE,
        ))];
        apply_filter_chain(&mut only_floor, &floor, &ctx()).expect("floor");
        assert_eq!(only_floor.len(), 1);
    }

    #[test]
    fn classification_is_soft_with_min_remaining() {
        let f = AccuracyPreferenceFilter {
            min_remaining: 2,
            bound: DEFAULT_ACCURACY_BOUND,
        };
        assert_eq!(f.classification(), FilterClass::Soft { min_remaining: 2 });
    }

    #[test]
    fn min_remaining_drives_chain_skip_decision() {
        let claimed = precision(false, Some(1e-5), None, None);
        let mut set = AlternativeSet::from_candidates(vec![
            vulkan(precision(false, None, None, None), 10),
            vulkan(claimed, 20),
        ]);
        let filters: Vec<Box<dyn AlternativeFilter>> = vec![Box::new(AccuracyPreferenceFilter {
            min_remaining: 2,
            bound: DEFAULT_ACCURACY_BOUND,
        })];
        apply_filter_chain(&mut set, &filters, &ctx()).expect("soft skip succeeds");
        assert_eq!(set.len(), 2, "below min_remaining: filter skipped");
    }
}

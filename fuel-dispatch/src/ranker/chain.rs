//! Filter-chain pipeline: applies an ordered list of
//! [`AlternativeFilter`]s to an [`AlternativeSet`], respecting each
//! filter's [`FilterClass`].
//!
//! Phase 1.1 of the picker-work arc.
//!
//! # Semantics
//!
//! For each filter in order:
//!
//! - Call `filter.filter(set.alternatives(), ctx)` to get the
//!   keep-mask (indices).
//! - If the filter is [`FilterClass::Hard`] and the keep-mask is
//!   empty, return [`Error::FilterRejected`] — the user asked for
//!   something the binding-table can't deliver and we surface it
//!   rather than silently substituting.
//! - If the filter is [`FilterClass::Soft { min_remaining }`] and
//!   the keep-mask would leave fewer than `min_remaining` entries,
//!   skip this filter for this call (it's a preference, not a
//!   requirement).
//! - Otherwise apply the keep-mask via
//!   [`AlternativeSet::retain_indices`].
//!
//! Filters operate over surviving candidates only — a hard filter at
//! position 3 sees what positions 1 and 2 left behind.

use fuel_core_types::{Error, Result};

use super::alternative_set::AlternativeSet;
use super::filter::{AlternativeFilter, FilterClass, FilterContext};

/// Apply an ordered chain of filters to `set`.
///
/// Returns `Ok(())` if every filter passed (or was soft-skipped).
/// Returns [`Error::FilterRejected`] if any hard filter rejected
/// every candidate.
pub fn apply_filter_chain(
    set: &mut AlternativeSet,
    filters: &[Box<dyn AlternativeFilter>],
    ctx: &FilterContext,
) -> Result<()> {
    for filter in filters {
        let keep_indices = filter.filter(set.alternatives(), ctx);
        let kept = keep_indices.len();

        match filter.classification() {
            FilterClass::Hard if kept == 0 => {
                return Err(Error::FilterRejected {
                    filter: filter.name(),
                    ctx_summary: ctx.summary(),
                    available_alternatives: set.len(),
                });
            }
            FilterClass::Soft { min_remaining } if kept < min_remaining => {
                // Filter would over-restrict; preference yields to
                // available diversity. Logged at debug so the call
                // site can introspect why N alternatives survived
                // when a preference "should have" narrowed further.
                tracing::debug!(
                    filter = filter.name(),
                    kept,
                    min_remaining,
                    "ranker: soft filter saturated, skipping",
                );
                continue;
            }
            _ => {}
        }

        set.retain_indices(&keep_indices);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::{CostEstimate, PrecisionGuarantee};
    use crate::kernel::{KernelCaps, OpParams};
    use crate::ranker::candidate::Candidate;

    use fuel_core_types::dispatch::OpKind;
    use fuel_core_types::probe::BackendId;
    use fuel_core_types::{DType, DeviceLocation, Layout, Result as FuelResult, Shape};
    use fuel_storage::Storage;
    use std::sync::{Arc, RwLock};

    fn noop(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> FuelResult<()> {
        Ok(())
    }

    fn dummy_candidate(flops: u64) -> Candidate {
        Candidate {
            kernel: noop,
            caps: KernelCaps::empty(),
            backend: BackendId::Cpu,
            device: DeviceLocation::Cpu,
            precision: PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            static_cost: CostEstimate { flops, bytes_moved: 0, kernel_overhead_ns: 0 },
            inbound_transfer_ns: 0,
            op_params: OpParams::None,
            coupling: Vec::new(),
            kernel_source: "",
        }
    }

    fn three_candidates() -> AlternativeSet {
        AlternativeSet::from_candidates(
            vec![dummy_candidate(1), dummy_candidate(2), dummy_candidate(3)],
            super::super::alternative_set::DEFAULT_MAX_N,
        )
    }

    /// Mock filter — returns a hand-coded keep-mask regardless of
    /// input, with configurable classification + name.
    struct StaticKeep {
        keep: Vec<usize>,
        class: FilterClass,
        name: &'static str,
    }
    impl AlternativeFilter for StaticKeep {
        fn filter(&self, _alts: &[Candidate], _ctx: &FilterContext) -> Vec<usize> {
            self.keep.clone()
        }
        fn classification(&self) -> FilterClass {
            self.class
        }
        fn name(&self) -> &'static str {
            self.name
        }
    }

    fn ctx() -> FilterContext<'static> {
        // 'static layouts via leak — only valid inside test bodies.
        static LAYOUTS: std::sync::OnceLock<Vec<Layout>> = std::sync::OnceLock::new();
        let layouts = LAYOUTS.get_or_init(|| {
            vec![Layout::contiguous(Shape::from(vec![4]))]
        });
        FilterContext::new(OpKind::AddElementwise, &[DType::F32], layouts)
    }

    #[test]
    fn empty_filter_list_is_identity() {
        let mut set = three_candidates();
        let filters: Vec<Box<dyn AlternativeFilter>> = Vec::new();
        apply_filter_chain(&mut set, &filters, &ctx()).expect("identity");
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn hard_filter_to_zero_returns_filter_rejected() {
        let mut set = three_candidates();
        let filters: Vec<Box<dyn AlternativeFilter>> = vec![Box::new(StaticKeep {
            keep: vec![],
            class: FilterClass::Hard,
            name: "drop-all-hard",
        })];
        let err = apply_filter_chain(&mut set, &filters, &ctx()).unwrap_err();
        match err {
            Error::FilterRejected { filter, available_alternatives, ctx_summary } => {
                assert_eq!(filter, "drop-all-hard");
                assert_eq!(available_alternatives, 3);
                assert!(ctx_summary.contains("AddElementwise"));
            }
            other => panic!("expected FilterRejected, got {other:?}"),
        }
        // Set is unchanged on rejection — caller may want to log it.
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn soft_filter_to_zero_is_skipped_set_unchanged() {
        let mut set = three_candidates();
        let filters: Vec<Box<dyn AlternativeFilter>> = vec![Box::new(StaticKeep {
            keep: vec![],
            class: FilterClass::Soft { min_remaining: 1 },
            name: "drop-all-soft",
        })];
        apply_filter_chain(&mut set, &filters, &ctx()).expect("soft skip succeeds");
        assert_eq!(
            set.len(), 3,
            "soft filter that would empty the set must be skipped",
        );
    }

    #[test]
    fn soft_filter_below_min_remaining_is_skipped() {
        let mut set = three_candidates();
        // Returns only 1 candidate, but min_remaining=2 → skip.
        let filters: Vec<Box<dyn AlternativeFilter>> = vec![Box::new(StaticKeep {
            keep: vec![0],
            class: FilterClass::Soft { min_remaining: 2 },
            name: "narrow-too-much",
        })];
        apply_filter_chain(&mut set, &filters, &ctx()).expect("soft skip succeeds");
        assert_eq!(
            set.len(), 3,
            "soft filter that would leave fewer than min_remaining must be skipped",
        );
    }

    #[test]
    fn soft_filter_at_or_above_min_remaining_applies() {
        let mut set = three_candidates();
        // Returns 2 candidates, min_remaining=2 → exactly at threshold, applies.
        let filters: Vec<Box<dyn AlternativeFilter>> = vec![Box::new(StaticKeep {
            keep: vec![0, 2],
            class: FilterClass::Soft { min_remaining: 2 },
            name: "narrow-just-enough",
        })];
        apply_filter_chain(&mut set, &filters, &ctx()).expect("apply");
        assert_eq!(set.len(), 2);
        let flops: Vec<u64> = set.alternatives().iter().map(|c| c.static_cost.flops).collect();
        assert_eq!(flops, vec![1, 3], "kept candidates at indices 0 and 2");
    }

    #[test]
    fn hard_filter_with_partial_keep_applies() {
        let mut set = three_candidates();
        let filters: Vec<Box<dyn AlternativeFilter>> = vec![Box::new(StaticKeep {
            keep: vec![1, 2],
            class: FilterClass::Hard,
            name: "narrow-hard",
        })];
        apply_filter_chain(&mut set, &filters, &ctx()).expect("apply");
        assert_eq!(set.len(), 2);
        let flops: Vec<u64> = set.alternatives().iter().map(|c| c.static_cost.flops).collect();
        assert_eq!(flops, vec![2, 3]);
    }

    #[test]
    fn filter_order_matters_hard_after_soft_sees_filtered_set() {
        let mut set = three_candidates();
        let filters: Vec<Box<dyn AlternativeFilter>> = vec![
            Box::new(StaticKeep {
                keep: vec![0, 1], // drops index 2 (flops=3)
                class: FilterClass::Soft { min_remaining: 1 },
                name: "soft-narrow",
            }),
            Box::new(StaticKeep {
                // After soft pass: set has flops [1, 2] at indices 0, 1.
                // Hard filter asks to keep index 0 only.
                keep: vec![0],
                class: FilterClass::Hard,
                name: "hard-pick-one",
            }),
        ];
        apply_filter_chain(&mut set, &filters, &ctx()).expect("apply");
        assert_eq!(set.len(), 1);
        assert_eq!(set.winner().unwrap().static_cost.flops, 1);
    }

    #[test]
    fn hard_first_fails_does_not_invoke_subsequent_filters() {
        let mut set = three_candidates();
        // If the chain didn't short-circuit, the second filter would
        // panic. We verify it never runs.
        struct Panicking;
        impl AlternativeFilter for Panicking {
            fn filter(&self, _: &[Candidate], _: &FilterContext) -> Vec<usize> {
                panic!("should not be invoked: hard filter ahead returned empty");
            }
            fn classification(&self) -> FilterClass {
                FilterClass::Soft { min_remaining: 1 }
            }
            fn name(&self) -> &'static str {
                "panicking"
            }
        }
        let filters: Vec<Box<dyn AlternativeFilter>> = vec![
            Box::new(StaticKeep {
                keep: vec![],
                class: FilterClass::Hard,
                name: "fail-fast",
            }),
            Box::new(Panicking),
        ];
        let err = apply_filter_chain(&mut set, &filters, &ctx()).unwrap_err();
        assert!(matches!(err, Error::FilterRejected { .. }));
    }

    #[test]
    fn soft_skip_does_not_block_subsequent_filters() {
        let mut set = three_candidates();
        let filters: Vec<Box<dyn AlternativeFilter>> = vec![
            // Soft would over-restrict → skipped, set stays at 3.
            Box::new(StaticKeep {
                keep: vec![],
                class: FilterClass::Soft { min_remaining: 1 },
                name: "soft-saturated",
            }),
            // Hard then narrows the still-full set.
            Box::new(StaticKeep {
                keep: vec![2],
                class: FilterClass::Hard,
                name: "hard-pick-last",
            }),
        ];
        apply_filter_chain(&mut set, &filters, &ctx()).expect("apply");
        assert_eq!(set.len(), 1);
        assert_eq!(set.winner().unwrap().static_cost.flops, 3);
    }

    #[test]
    fn empty_set_with_hard_filter_returns_zero_alternatives_summary() {
        let mut set = AlternativeSet::empty();
        let filters: Vec<Box<dyn AlternativeFilter>> = vec![Box::new(StaticKeep {
            keep: vec![],
            class: FilterClass::Hard,
            name: "any-hard",
        })];
        let err = apply_filter_chain(&mut set, &filters, &ctx()).unwrap_err();
        match err {
            Error::FilterRejected { available_alternatives, .. } => {
                assert_eq!(available_alternatives, 0);
            }
            other => panic!("expected FilterRejected, got {other:?}"),
        }
    }
}
